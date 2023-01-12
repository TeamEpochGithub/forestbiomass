import torch
from PIL import Image
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import os
import rasterio
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import warnings
import numpy as np
import os.path as osp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

import models
import data
import csv
from models.utils import loss_functions
import argparse
import models.utils.transforms as tf
from models.utils.dataloading import SentinelTiffDataloader, SentinelTiffDataloaderSubmission, create_tensor, \
    apply_transforms
import operator
import sys

from models.utils.simple_tensor_accumulate import accumulate_predictions

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Sentinel2Model(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.learning_rate = args.learning_rate
        self.train_loss_function = args.train_loss_function
        self.val_loss_function = args.val_loss_function

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.train_loss_function(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.val_loss_function(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        return self.model(x)


def prepare_dataset_training(args):
    with open(args.training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    training_features_path = args.tiff_training_features_path

    id_month_list = []

    for current_id in chip_ids:
        for month in range(0, 12):

            if month < 10:
                month = "0" + str(month)

            month_patch_path = osp.join(training_features_path, f"{current_id}_S2_{month}.tif")
            if osp.exists(month_patch_path):
                id_month_list.append((current_id, month))

    corrupted_transform_method, transform_channels = tf.select_transform_method(args.transform_method,
                                                                                in_channels=len(
                                                                                    args.bands_to_keep))

    new_dataset = SentinelTiffDataloader(training_features_path,
                                         args.tiff_training_labels_path,
                                         id_month_list,
                                         args.bands_to_keep,
                                         corrupted_transform_method)

    return new_dataset


def prepare_dataset_testing(args) -> Dataset:
    with open(args.testing_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    testing_features_path = args.tiff_testing_features_path

    id_month_list = []

    for current_id in chip_ids:
        for month in range(0, 12):

            if month < 10:
                month = "0" + str(month)

            month_patch_path = osp.join(testing_features_path, f"{current_id}_S2_{month}.tif")
            if osp.exists(month_patch_path):
                id_month_list.append((current_id, month))

    corrupted_transform_method, transform_channels = tf.select_transform_method(args.transform_method,
                                                                                in_channels=len(
                                                                                    args.bands_to_keep))

    new_dataset = SentinelTiffDataloaderSubmission(testing_features_path,
                                                   id_month_list,
                                                   args.bands_to_keep,
                                                   corrupted_transform_method)
    return new_dataset, id_month_list


def select_segmenter(encoder_weights, segmenter_name, encoder_name, number_of_channels):
    if segmenter_name == "Unet":
        base_model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=number_of_channels,
            classes=1,
            encoder_weights=encoder_weights
        )
    else:
        base_model = None

    assert base_model is not None, "Segmenter name was not recognized."
    return base_model


def train(args):
    print('=' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=' * 30)

    print("Getting train data...")

    train_dataset = prepare_dataset_training(args)

    train_size = int(1 - args.validation_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)

    base_model = select_segmenter(args.encoder_weights, args.segmenter_name, args.encoder_name, len(args.bands_to_keep))

    model = Sentinel2Model(base_model, args)

    logger = TensorBoardLogger("tb_logs", name=args.model_identifier)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="val/loss",
        mode="min",
    )

    # ddp = DDPStrategy(process_group_backend="gloo")
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=[logger],
        log_every_n_steps=args.log_step_frequency,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        # num_nodes=4,
        # strategy=ddp
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    return model, str(trainer.callback_metrics['val/loss'].item())


def load_model(args):
    print("Getting saved model...")

    assert osp.exists(args.current_model_path) is True, "requested model does not exist"
    log_folder_path = args.current_model_path

    version_dir = list(os.scandir(log_folder_path))[args.model_version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")
    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]
    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    base_model = select_segmenter(args.encoder_weights, args.segmenter_name, args.encoder_name,
                                  len(args.bands_to_keep) + args.extra_channels)

    # This block might be redundant if we can download weights via the python segmentation models library.
    # However, it might be that not all weights are available this way.
    # If you have downloaded weights (in the .pt format), put them in the pre-trained-weights folder
    # and give the file the same name as the encoder you're using.
    # If you do that, this block will try and load them for your model.
    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    ###########################################################

    model = Sentinel2Model(base_model, args)

    checkpoint = torch.load(str(latest_checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    print("Model loaded")
    return model


def create_submissions(args):

    model = load_model(args)

    new_dataset, id_month_list = prepare_dataset_testing(args)

    trainer = Trainer(accelerator="gpu", devices=1)

    dl = DataLoader(new_dataset, num_workers=18)

    predictions = trainer.predict(model, dataloaders=dl)
    tensor_id_list = [i[0] for i in id_month_list]

    transformed_predictions = [x.cpu().squeeze().detach().numpy() for x in predictions]
    linked_tensor_list = list(zip(tensor_id_list, transformed_predictions))
    linked_tensor_list = sorted(linked_tensor_list, key=operator.itemgetter(0))

    averaged_tensor_list = list(accumulate_predictions(linked_tensor_list))

    for id_tensor_pair in averaged_tensor_list:
        current_id = id_tensor_pair[0]
        current_tensor = id_tensor_pair[1]

        agbm_path = osp.join(args.submission_folder_path, f"{current_id}_agbm.tif")

        im = Image.fromarray(current_tensor)
        im.save(agbm_path)

    print("Finished creating submission.")


def set_args():
    model_segmenter = "Unet"
    model_encoder = "efficientnet-b2"
    model_encoder_weights = "imagenet"  # Leave None if not using weights.
    data_type = "tiff"  # options are "npy" or "tiff"
    epochs = 1
    learning_rate = 1e-4
    dataloader_workers = 1
    validation_fraction = 0.2
    batch_size = 1
    log_step_frequency = 200
    version = -1  # Keep -1 if loading the latest model version.
    save_top_k_checkpoints = 3
    transform_method = "replace_corrupted_0s"  # "replace_corrupted_noise"  # nothing  # add_band_corrupted_arrays
    train_loss_function = loss_functions.rmse_loss
    val_loss_function = loss_functions.rmse_loss

    # WARNING: Only increment extra_channels when making predictions/submission (based on the transform method used)
    # it is automatically incremented during training based on the transform method used (extra channels generated)
    extra_channels = 0

    band_map = {
        # S2 bands
        0: 'S2-B2: Blue-10m',
        1: 'S2-B3: Green-10m',
        2: 'S2-B4: Red-10m',
        3: 'S2-B5: VegRed-704nm-20m',
        4: 'S2-B6: VegRed-740nm-20m',
        5: 'S2-B7: VegRed-780nm-20m',
        6: 'S2-B8: NIR-833nm-10m',
        7: 'S2-B8A: NarrowNIR-864nm-20m',
        8: 'S2-B11: SWIR-1610nm-20m',
        9: 'S2-B12: SWIR-2200nm-20m',
        10: 'S2-CLP: CloudProb-160m',
        # S1 bands
        11: 'S1-VV-Asc: Cband-10m',
        12: 'S1-VH-Asc: Cband-10m',
        13: 'S1-VV-Desc: Cband-10m',
        14: 'S1-VH-Desc: Cband-10m',
        # Bands derived by transforms
        15: 'S2-NDVI: (NIR-Red)/(NIR+Red) 10m',
        16: 'S1-NDVVVH-Asc: Norm Diff VV & VH, 10m',
        17: 'S2-NDBI: Difference Built-up Index, 20m',
        18: 'S2-NDRE: Red Edge Vegetation Index, 20m',
        19: 'S2-NDSI: Snow Index, 20m',
        20: 'S2-NDWI: Water Index, 10m',
        21: 'S2-SWI: Sandardized Water-Level Index, 20m',
        22: 'S1-VV/VH-Asc: Cband-10m',
        23: 'S2-VV/VH-Desc: Cband-10m'
    }

    bands_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    band_indicator = ["1" if k in bands_to_keep else "0" for k, v in band_map.items()]

    parser = argparse.ArgumentParser()
    bands_to_keep_indicator = "bands-" + ''.join(str(x) for x in band_indicator)
    model_identifier = f"{model_segmenter}_{model_encoder}_{bands_to_keep_indicator}"

    parser.add_argument('--model_identifier', default=model_identifier, type=str)
    parser.add_argument('--segmenter_name', default=model_segmenter, type=str)
    parser.add_argument('--encoder_name', default=model_encoder, type=str)
    parser.add_argument('--encoder_weights', default=model_encoder_weights, type=str)
    parser.add_argument('--model_version', default=version, type=int)
    parser.add_argument('--data_type', default=data_type, type=str)

    data_path = osp.dirname(data.__file__)
    models_path = osp.dirname(models.__file__)

    parser.add_argument('--tiff_training_features_path', default=str(osp.join(data_path, "imgs", "train_features")))
    parser.add_argument('--tiff_training_labels_path', default=str(osp.join(data_path, "imgs", "train_agbm")))
    parser.add_argument('--tiff_testing_features_path', default=str(osp.join(data_path, "imgs", "test_features")))

    parser.add_argument('--training_ids_path', default=str(osp.join(data_path, "patch_names")), type=str)
    parser.add_argument('--testing_ids_path', default=str(osp.join(data_path, "test_patch_names")), type=str)
    parser.add_argument('--current_model_path', default=str(osp.join(models_path, "tb_logs", model_identifier)),
                        type=str)
    parser.add_argument('--submission_folder_path', default=str(osp.join(data_path, "imgs", "test_agbm")), type=str)

    parser.add_argument('--dataloader_workers', default=dataloader_workers, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--learning_rate', default=learning_rate, type=float)
    parser.add_argument('--validation_fraction', default=validation_fraction, type=float)
    parser.add_argument('--log_step_frequency', default=log_step_frequency, type=int)
    parser.add_argument('--save_top_k_checkpoints', default=save_top_k_checkpoints, type=int)

    parser.add_argument('--bands_to_keep', default=bands_to_keep, type=list)
    parser.add_argument('--train_loss_function', default=train_loss_function)
    parser.add_argument('--val_loss_function', default=val_loss_function)
    parser.add_argument('--transform_method', default=transform_method, type=str)
    parser.add_argument('--extra_channels', default=extra_channels, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = set_args()
    #_, score = train(args)
    # print(score)

    create_submissions(args)
