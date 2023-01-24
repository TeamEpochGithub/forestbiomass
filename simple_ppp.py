import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import os
import rasterio
import warnings
import numpy as np
import os.path as osp
import models
import data
import csv
from models.utils import loss_functions
import argparse
from tqdm import tqdm
import pandas as pd
from torch import nn
from torchgeo.transforms import indices
import models.utils.transforms as tf
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision('medium')

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels,latent_dim=128):
        super(Linear,self).__init__()
        self.linear = nn.Linear(in_channels, latent_dim)
        self.linear2 = nn.Linear(latent_dim, latent_dim//2)
        self.linear3 = nn.Linear(latent_dim//2, out_channels)
        self.batch_norm=nn.BatchNorm1d(in_channels)
        self.seq=nn.Sequential(self.linear, nn.ReLU(), self.linear2,nn.ReLU(),self.linear3,nn.Sigmoid())

    def forward(self, x):
        x=self.batch_norm(x)
        linear = self.seq(x)
        return linear

class PixelWiseNet(pl.LightningModule):
    def __init__(self, model,lr=0.001):
        super(PixelWiseNet, self).__init__()
        self.lr=lr
        self.model = model

    def forward(self, x):
        x=x.reshape(-1,x.shape[-1])
        linear = self.model(x)
        return linear

    def training_step(self,train_batch, batch_idx):
        x, y = train_batch
        x=x.reshape(-1,x.shape[-1])
        y=y.reshape(-1,y.shape[-1])
        # print(x.shape, y.shape)
        y_hat = self.model(x)
        # print(y)
        # print(y_hat)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss,sync_dist=True)
        self.log('train_rmse', torch.sqrt(loss)*256,sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x=x.reshape(-1,x.shape[-1])
        y=y.reshape(-1,y.shape[-1])
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss,sync_dist=True)
        self.log('val_rmse', torch.sqrt(loss)*256,sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x=x.reshape(-1,x.shape[-1])
        y=y.reshape(-1,y.shape[-1])
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss,sync_dist=True)
        self.log('test_rmse', torch.sqrt(loss)*256,sync_dist=True)
        return loss 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class ChainedSegmentationDatasetMaker(Dataset):
    def __init__(self, training_feature_path, training_labels_path, id_list, data_type, S1_bands, S2_bands,
                 month_selection, transform=None,test=False):
        self.training_feature_path = training_feature_path
        self.training_labels_path = training_labels_path
        self.id_list = id_list
        self.data_type = data_type
        self.S1_bands = S1_bands
        self.S2_bands = S2_bands
        self.transform = transform
        self.month_selection = month_selection
        self.test=test

    def __len__(self):
        return len(self.id_list)#*256**2

    def __getitem__(self, idx):

        id = self.id_list[idx]
        if self.test:
            label_tensor =  torch.rand((1,256,256))
        else:
            label_path = osp.join(self.training_labels_path, f"{id}_agbm.tif")
            label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        tensor_list = []
        for month_index, month_indicator in enumerate(self.month_selection):
            if month_indicator == 0:
                continue
            feature_tensor = retrieve_tiff(self.training_feature_path, id, str(month_index), self.S1_bands,
                                           self.S2_bands)#.to(torch.float16)
            # print(feature_tensor.shape)
            sample = {'image': feature_tensor, 'label': label_tensor}
            feature_tensor = apply_transforms()(sample)
            feature_tensor = feature_tensor['image']
            tensor_list.append(feature_tensor)

        tensor_list = torch.cat(tensor_list, dim=0)
        # print(label_tensor.shape, tensor_list.shape)
        tensor_list=tensor_list.reshape(tensor_list.shape[0],tensor_list.shape[1]*tensor_list.shape[2]).permute(1,0)/255
        label_tensor=label_tensor.reshape(label_tensor.shape[0],label_tensor.shape[1]*label_tensor.shape[2]).permute(1,0)/255
        # print(label_tensor.shape, tensor_list.shape)
        # s = idx // (256**2)
        # j = idx % (256**2)
        # tensor_list, label_tensor = tensor_list[s*256**2+j,:].squeeze(), label_tensor[s*256**2+j,:].squeeze()
        return tensor_list, label_tensor

def apply_transforms():
    return nn.Sequential(
        tf.ClampAGBM(vmin=0., vmax=500.),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
        indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
        indices.AppendNormalizedDifferenceIndex(index_a=11, index_b=12),  # (VV-VH)/(VV+VH), index 16
        indices.AppendNDBI(index_swir=8, index_nir=6),
        # Difference Built-up Index for development detection, index 17
        indices.AppendNDRE(index_nir=6, index_vre1=3),  # Red Edge Vegetation Index for canopy detection, index 18
        indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
        indices.AppendNDWI(index_green=1, index_nir=6),  # Difference Water Index for water detection, index 20
        indices.AppendSWI(index_vre1=3, index_swir2=8),)
        # tf.DropBands(torch.device('cpu'), [0,1,2,3,4,5,6,15,16,18,20]))

def create_tensor_from_bands_list(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)
    band_tensor = torch.tensor(band_array)
    # band_tensor/=255
    # # normalization happens here
    # band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (
    #     band_tensor.std(dim=(1, 2)) + 0.01
    # )
    # band_tensor = band_tensor.permute(2, 0, 1)
    return band_tensor

def retrieve_tiff(feature_path, id, month, S1_band_selection, S2_band_selection):
    if int(month) < 10:
        month = "0" + month

    channel_count = S1_band_selection.count(1) + S2_band_selection.count(1)
    S1_path = osp.join(feature_path, f"{id}_S1_{month}.tif")
    S2_path = osp.join(feature_path, f"{id}_S2_{month}.tif")
    if S2_band_selection.count(1) >= 1:
        if not osp.exists(S2_path):
            return create_tensor_from_bands_list(
                np.zeros((channel_count, 256, 256), dtype=np.float32)
            )

    bands = []
    if S1_band_selection.count(1) >= 1:
        S1_bands = rasterio.open(S1_path).read().astype(np.float32)
        S1_bands = [x for x, y in zip(S1_bands, S1_band_selection) if y == 1]
        bands.extend(S1_bands)

    if S2_band_selection.count(1) >= 1:
        S2_bands = rasterio.open(S2_path).read().astype(np.float32)
        S2_bands = [x for x, y in zip(S2_bands, S2_band_selection) if y == 1]
        bands.extend(S2_bands)
    feature_tensor = create_tensor_from_bands_list(bands)
    return feature_tensor

def prepare_dataset_training(args):
    with open(args.training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]


    training_features_path = args.tiff_training_features_path

    new_dataset = ChainedSegmentationDatasetMaker(training_features_path,
                                                  args.tiff_training_labels_path,
                                                  chip_ids,
                                                  args.data_type,
                                                  args.S1_band_selection,
                                                  args.S2_band_selection,
                                                  args.month_selection)

    return new_dataset


def set_args():
    band_segmenter = "Unet"
    band_encoder = "efficientnet-b2"
    band_encoder_weights = "imagenet"

    month_segmenter = "Unet"
    month_encoder = "efficientnet-b2"
    month_encoder_weights = "imagenet"

    data_type = "tiff"  # options are "npy" or "tiff"
    epochs = 50
    learning_rate = 1e-3
    dataloader_workers = 10
    validation_fraction = 0.2
    batch_size = 4
    log_step_frequency = 10
    version = -1  # Keep -1 if loading the latest model version.
    save_top_k_checkpoints = 1
    loss_function = loss_functions.rmse_loss

    missing_month_repair_mode = "zeros"

    month_selection = {
        "September": 1,
        "October": 0,
        "November": 0,
        "December": 0,
        "January": 0,
        "February": 0,
        "March": 0,
        "April": 1,
        "May": 1,
        "June": 1,
        "July": 1,
        "August": 1,
    }
    # month_selection = {
    #     "September": 1,
    #     "October": 1,
    #     "November": 1,
    #     "December": 1,
    #     "January": 1,
    #     "February": 1,
    #     "March": 1,
    #     "April": 1,
    #     "May": 1,
    #     "June": 1,
    #     "July": 1,
    #     "August": 1,
    # }

    month_list = list(month_selection.values())

    month_selection_indicator = "months-" + "".join(str(x) for x in month_list)

    sentinel_1_bands = {
        "VV ascending": 1,
        "VH ascending": 1,
        "VV descending": 1,
        "VH descending": 1,
    }

    sentinel_2_bands = {
        "B2-Blue": 1,
        "B3-Green": 1,
        "B4-Red": 1,
        "B5-Veg red edge 1": 1,
        "B6-Veg red edge 2": 1,
        "B7-Veg red edge 3": 1,
        "B8-NIR": 1,
        "B8A-Narrow NIR": 1,
        "B11-SWIR 1": 1,
        "B12-SWIR 2": 1,
        "Cloud probability": 1,
    }

    s1_list = list(sentinel_1_bands.values())
    s2_list = list(sentinel_2_bands.values())

    s1_bands_indicator = "S1-" + "".join(str(x) for x in s1_list)
    s2_bands_indicator = "S2-" + "".join(str(x) for x in s2_list)

    parser = argparse.ArgumentParser()

    if band_encoder_weights is not None and month_encoder_weights is not None:
        model_identifier = (
            f"Bands_{band_segmenter}_{band_encoder}_{band_encoder_weights}_{s1_bands_indicator}_{s2_bands_indicator}"
            f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"
        )

    elif band_encoder_weights is None and month_encoder_weights is not None:
        model_identifier = (
            f"Bands_{band_segmenter}_{band_encoder}_{s1_bands_indicator}_{s2_bands_indicator}"
            f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"
        )

    elif band_encoder_weights is not None and month_encoder_weights is None:
        model_identifier = (
            f"Bands_{band_segmenter}_{band_encoder}_{s1_bands_indicator}_{s2_bands_indicator}"
            f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"
        )
    else:
        model_identifier = (
            f"Bands_{band_segmenter}_{band_encoder}_{s1_bands_indicator}_{s2_bands_indicator}"
            f"_Months_{month_segmenter}_{month_encoder}_{month_selection_indicator}"
        )

    parser.add_argument("--model_identifier", default=model_identifier, type=str)

    parser.add_argument("--band_segmenter_name", default=band_segmenter, type=str)
    parser.add_argument("--band_encoder_name", default=band_encoder, type=str)
    parser.add_argument(
        "--band_encoder_weights_name", default=band_encoder_weights, type=str
    )

    parser.add_argument("--month_segmenter_name", default=month_segmenter, type=str)
    parser.add_argument("--month_encoder_name", default=month_encoder, type=str)
    parser.add_argument(
        "--month_encoder_weights_name", default=month_encoder_weights, type=str
    )

    parser.add_argument("--model_version", default=version, type=int)
    parser.add_argument("--data_type", default=data_type, type=str)

    data_path = "C:/Users/kuipe/Desktop/Epoch/forestbiomass/data"
    models_path = osp.dirname(models.__file__)

    parser.add_argument(
        "--tiff_training_features_path",
        default=str(osp.join(data_path, "imgs", "train_features")),
    )
    parser.add_argument(
        "--tiff_training_labels_path",
        default=str(osp.join(data_path, "imgs", "train_agbm")),
    )
    parser.add_argument(
        "--tiff_testing_features_path",
        default=str(osp.join(data_path, "imgs", "test_features")),
    )

    parser.add_argument(
        "--training_ids_path", default=str(osp.join(data_path, "patch_names")), type=str
    )
    parser.add_argument(
        "--testing_ids_path",
        default=str(osp.join(data_path, "test_patch_names")),
        type=str,
    )

    parser.add_argument(
        "--current_model_path",
        default=str(osp.join(models_path, "tb_logs", model_identifier)),
        type=str,
    )
    parser.add_argument(
        "--submission_folder_path",
        default=str(osp.join(data_path, "imgs", "test_agbm")),
        type=str,
    )

    parser.add_argument("--dataloader_workers", default=dataloader_workers, type=int)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--learning_rate", default=learning_rate, type=float)
    parser.add_argument(
        "--validation_fraction", default=validation_fraction, type=float
    )
    parser.add_argument("--log_step_frequency", default=log_step_frequency, type=int)
    parser.add_argument(
        "--save_top_k_checkpoints", default=save_top_k_checkpoints, type=int
    )

    parser.add_argument("--S1_band_selection", default=s1_list, type=list)
    parser.add_argument("--S2_band_selection", default=s2_list, type=list)
    parser.add_argument("--month_selection", default=month_list, type=list)
    parser.add_argument("--loss_function", default=loss_function)

    parser.add_argument(
        "--missing_month_repair_mode", default=missing_month_repair_mode, type=str
    )
    parser.add_argument("--in_channels", default=132, type=int)#264,132

    args = parser.parse_args()

    print("=" * 30)
    for arg in vars(args):
        print("--", arg, ":", getattr(args, arg))
    print("=" * 30)

    return args

def train(args,train_dataloader,valid_dataloader):
    logger=WandbLogger(project="forestbiomass",name="pixelwise_allbands")
    tb_logger = TensorBoardLogger("tb_logs", name="pixelwise_pytorch")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="val_rmse",
        mode="min",
    )
    linear_model=Linear(in_channels=args.in_channels,out_channels=1)
    model=PixelWiseNet(linear_model,lr=args.learning_rate)
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=args.epochs,
        logger=[tb_logger,logger],
        log_every_n_steps=args.log_step_frequency,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        strategy=DDPStrategy(process_group_backend="gloo",find_unused_parameters=False),
    )

    trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    args = set_args()
    train_dataset = prepare_dataset_training(args)
    train_size = int((1 - args.validation_fraction) * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,num_workers=args.dataloader_workers)
    save_path = 'pixelwise_pytorch'  # specifies folder to store trained models
    train(args,train_dataloader,val_dataloader)