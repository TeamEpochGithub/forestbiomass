import sys

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import distributed as dist
import segmentation_models_pytorch as smp
import os
import rasterio
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import warnings
import numpy as np
import os.path as osp
from pytorch_lightning.loggers import TensorBoardLogger
import data
import models
import csv
import argparse

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class Segmentation_Dataset_Maker(Dataset):
    def __init__(self, training_data_path, id_month_list, s1_bands, s2_bands, transform=None):
        self.training_data_path = training_data_path
        self.id_month_list = id_month_list
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.transform = transform

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):

        id, month = self.id_month_list[idx]

        label_path = osp.join(self.training_data_path, id, "label.npy")
        label = np.load(label_path, allow_pickle=True)
        label_tensor = torch.tensor(np.asarray([label], dtype=np.float32))

        arr_list = []

        for index, s1_index in enumerate(self.s1_bands):

            if s1_index == 1:
                band = np.load(osp.join(self.training_data_path, id, month, "S1", f"{index}.npy"), allow_pickle=True)
                arr_list.append(band)

        for index, s2_index in enumerate(self.s2_bands):

            if s2_index == 1:
                band = np.load(osp.join(self.training_data_path, id, month, "S2", f"{index}.npy"), allow_pickle=True)
                arr_list.append(band)

        data_tensor = torch.tensor(np.asarray(arr_list, dtype=np.float32))

        data_tensor = (data_tensor.permute(1, 2, 0) - data_tensor.mean(dim=(1, 2))) / (
                    data_tensor.std(dim=(1, 2)) + 0.01)
        data_tensor = data_tensor.permute(2, 0, 1)

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor


class Sentinel2Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss)
        self.log("train/rmse", torch.sqrt(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train/loss", loss)
        self.log("train/rmse", torch.sqrt(loss))
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.0001)]

    def forward(self, x):
        return self.model(x)


def prepare_dataset(args):

    with open(args.training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    id_month_list = []

    for id in chip_ids:

        for month in range(0, 12):

            month_patch_path = osp.join(args.training_features_path, id, str(month))  # 1 is the green band, out of the 11 bands

            if osp.exists(osp.join(month_patch_path, "S2")):

                id_month_list.append((id, str(month)))

    new_dataset = Segmentation_Dataset_Maker(args.training_features_path, id_month_list, args.S1_band_selection, args.S2_band_selection)
    return new_dataset, (len(args.S1_band_selection) + len(args.S2_band_selection))


def select_segmenter(segmenter_name, encoder_name, number_of_channels):

    if segmenter_name == "Unet":

        base_model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=number_of_channels,
            classes=1
        )

    else:
        base_model = None

    assert base_model is not None, "Segmenter name was not recognized."

    return base_model


def create_tensor(band_list):

    band_list = np.asarray(band_list, dtype=np.float32)

    band_tensor = torch.tensor(band_list)
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (band_tensor.std(dim=(1, 2)) + 0.01)
    band_tensor = band_tensor.permute(2, 0, 1)

    return band_tensor.unsqueeze(0)


def train(args):

    print("Getting train data...")

    train_dataset, number_of_channels = prepare_dataset(args)

    train_size = int(1 - args.validation_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_workers)

    base_model = select_segmenter(args.segmenter_name, args.encoder_name, number_of_channels)

    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    s2_model = Sentinel2Model(base_model)

    logger = TensorBoardLogger("tb_logs", name=args.model_identifier)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=[logger],
        log_every_n_steps=5,
    )

    trainer.fit(s2_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    return s2_model


def load_model(segmenter_name, encoder_name, number_of_channels, version=None):

    print("Getting saved model...")

    log_folder_path = osp.join(osp.dirname(models.__file__), "tb_logs", f"{segmenter_name}_{encoder_name}")

    if version is None:
        version_dir = list(os.scandir(log_folder_path))[-1]
    else:
        version_dir = list(os.scandir(log_folder_path))[version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")

    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]

    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    base_model = select_segmenter(segmenter_name, encoder_name, number_of_channels)

    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    s2_model = Sentinel2Model(base_model)

    checkpoint = torch.load(latest_checkpoint_path)
    s2_model.load_state_dict(checkpoint["state_dict"])

    return s2_model


def loading_example():
    model = load_model("Unet", "resnet50", 10)

    example_path = osp.join(osp.dirname(data.__file__), "converted", "2d977c85", "0", "S2")

    collected_bands = []

    for i in range(0, 10):
        collected_bands.append(np.load(osp.join(example_path, f"{i}.npy"), allow_pickle=True))

    model_input = create_tensor(collected_bands)

    prediction = model(model_input)

    plt.imshow(prediction.cpu().squeeze().detach().numpy(), interpolation='nearest')
    plt.show()


def create_submissions():

    model = load_model("Unet", "efficientnet-b7", 14)

    test_data_path = r"\\DESKTOP-P8NCSTN\Epoch\forestbiomass\data\testing_converted"

    with open(osp.join(osp.dirname(data.__file__), 'test_patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    total = len(patch_names)

    for index, id in enumerate(patch_names):

        all_months = []

        for month in range(0, 12):

            if month < 10:
                month = f"0{month}"
            else:
                month = f"{month}"

            s1_folder_path = osp.join(test_data_path, id, str(month), "S1")
            s2_folder_path = osp.join(test_data_path, id, str(month), "S2")

            if osp.exists(s2_folder_path):

                all_bands = []

                for s1_index in range(0, 4):

                    band = np.load(osp.join(s1_folder_path, f"{s1_index}.npy"), allow_pickle=True)
                    all_bands.append(band)

                for s2_index in range(0, 10):

                    band = np.load(osp.join(s2_folder_path, f"{s2_index}.npy"), allow_pickle=True)
                    all_bands.append(band)

                input_tensor = torch.tensor(np.asarray(all_bands, dtype=np.float32))

                input_tensor = (input_tensor.permute(1, 2, 0) - input_tensor.mean(dim=(1, 2))) / (input_tensor.std(dim=(1, 2)) + 0.01)
                input_tensor = input_tensor.permute(2, 0, 1)
                input_tensor = input_tensor.unsqueeze(0)

                pred = model(input_tensor)

                pred = pred.cpu().squeeze().detach().numpy()

                all_months.append(pred)

        count = len(all_months)

        agbm_arr = np.asarray(sum(all_months) / count)

        test_agbm_path = osp.join(osp.dirname(data.__file__), "imgs", "test_agbm", f"{id}_agbm.tif")

        im = Image.fromarray(agbm_arr)
        im.save(test_agbm_path)

        if index % 100 == 0:
            print(f"{index} / {total}")


def set_args():

    model_segmenter = "Unet"
    model_encoder = "efficientnet-b7"
    epochs = 20
    learning_rate = 1e-4
    dataloader_workers = 6
    validation_fraction = 0.2
    batch_size = 8

    sentinel_1_bands = {
        "VV ascending": 0,
        "VH ascending": 0,
        "VV descending": 0,
        "VH descending": 0
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
        "Cloud probability": 0
    }

    s1_list = list(sentinel_1_bands.values())
    s2_list = list(sentinel_2_bands.values())

    s1_bands_indicator = "S1-" + ''.join(str(x) for x in s1_list)
    s2_bands_indicator = "S2-" + ''.join(str(x) for x in s2_list)

    parser = argparse.ArgumentParser()

    model_identifier = f"{model_segmenter}_{model_encoder}_{s1_bands_indicator}_{s2_bands_indicator}"
    parser.add_argument('--model_identifier', default=model_identifier, type=str)
    parser.add_argument('--segmenter_name', default=model_segmenter, type=str)
    parser.add_argument('--encoder_name', default=model_encoder, type=str)

    data_path = osp.dirname(data.__file__)
    models_path = osp.dirname(models.__file__)
    parser.add_argument('--training_features_path', default=str(osp.join(data_path, "converted")), type=str)
    parser.add_argument('--training_ids_path', default=str(osp.join(data_path, "patch_names")), type=str)
    parser.add_argument('--testing_features_path', default=str(osp.join(data_path, "testing_converted")), type=str)
    parser.add_argument('--testing_ids_path', default=str(osp.join(data_path, "test_patch_names")), type=str)
    parser.add_argument('--current_model_path', default=str(osp.join(models_path, model_identifier)), type=str)

    parser.add_argument('--dataloader_workers', default=dataloader_workers, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--learning_rate', default=learning_rate, type=float)
    parser.add_argument('--validation_fraction', default=validation_fraction, type=float)

    parser.add_argument('--S1_band_selection', default=s1_list, type=list)
    parser.add_argument('--S2_band_selection', default=s2_list, type=list)

    args = parser.parse_args()

    print('=' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=' * 30)

    return args

if __name__ == '__main__':
    args = set_args()

    print(args.epochs)

    #create_submissions()
    #train("Unet", "efficientnet-b7", 100 , 0.8)

    # No more bad paths
    # Maybe more functions
    # Related code closer together
    # Review whitespace usage
    # Class names