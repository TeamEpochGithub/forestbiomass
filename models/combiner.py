import sys

import torch
from PIL import Image
import numpy
from torch.utils.data import Dataset, DataLoader
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
from pytorch_lightning.callbacks import ModelCheckpoint
import models
import data
import csv
from models.utils import loss_functions
from pytorch_lightning.strategies import DDPStrategy
import argparse

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class CombinationDatasetMaker(Dataset):
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

        data_tensor = create_tensor(arr_list)

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor


class CombinationDatasetMakerTIFF(Dataset):
    def __init__(self, training_feature_path, training_label_path, id_month_list, S1_bands, S2_bands, transform=None):
        self.training_feature_path = training_feature_path
        self.training_label_path = training_label_path
        self.id_month_list = id_month_list
        self.S1_bands = S1_bands
        self.S2_bands = S2_bands
        self.transform = transform

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):

        id, month = self.id_month_list[idx]

        label_path = osp.join(self.training_label_path, f"{id}_agbm.tif")
        label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        if int(month) < 10:
            month = "0" + month

        S1_data_path = osp.join(self.training_feature_path, f"{id}_S1_{month}.tif")
        S2_data_path = osp.join(self.training_feature_path, f"{id}_S2_{month}.tif")

        bands = []

        if self.S1_bands.count(1) >= 1:
            S1_bands = rasterio.open(S1_data_path).read().astype(np.float32)
            S1_bands = [x for x, y in zip(S1_bands, self.S1_bands) if y == 1]
            bands.extend(S1_bands)

        if self.S2_bands.count(1) >= 1:
            S2_bands = rasterio.open(S2_data_path).read().astype(np.float32)
            S2_bands = [x for x, y in zip(S2_bands, self.S2_bands) if y == 1]
            bands.extend(S2_bands)

        feature_tensor = create_tensor(bands)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        return feature_tensor, label_tensor


class Combiner(pl.LightningModule):
    def __init__(self, model, learning_rate, loss_function):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        return self.model(x)


def prepare_dataset(training_ids_path, training_features_path, S1_band_selection, S2_band_selection):
    with open(training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    id_month_list = []
    for id in chip_ids:
        for month in range(0, 12):

            month_patch_path = osp.join(training_features_path, id, str(month))
            if osp.exists(osp.join(month_patch_path, "S2")):
                id_month_list.append((id, str(month)))

    new_dataset = CombinationDatasetMaker(training_features_path, id_month_list, S1_band_selection,
                                          S2_band_selection)
    return new_dataset, (S1_band_selection.count(1) + S2_band_selection.count(1))


def prepare_dataset_tiff(training_ids_path, training_features_path, S1_band_selection, S2_band_selection):
    with open(training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    id_month_list = []
    for id in chip_ids:
        for month in range(0, 12):

            month_patch_path = osp.join(training_features_path, "train_features", f"{id}_S2_{month:02}.tif")
            if osp.exists(month_patch_path):
                id_month_list.append((id, str(month)))

    features_path = osp.join(training_features_path, "train_features")
    labels_path = osp.join(training_features_path, "train_agbm")

    new_dataset = CombinationDatasetMakerTIFF(features_path, labels_path, id_month_list, S1_band_selection,
                                              S2_band_selection)
    return new_dataset, (S1_band_selection.count(1) + S2_band_selection.count(1))


def select_segmenter(args):
    channel_count = (args.S1_band_selection.count(1) + args.S2_band_selection.count(1))

    if args.segmenter_name == "Unet":

        base_model = smp.Unet(
            encoder_name=args.encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=args.encoder_weights
        )
    else:
        base_model = None

    assert base_model is not None, "Segmenter name was not recognized."
    return base_model


def create_tensor(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)

    band_tensor = torch.tensor(band_array)

    # normalization happens here
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (band_tensor.std(dim=(1, 2)) + 0.01)
    band_tensor = band_tensor.permute(2, 0, 1)

    return band_tensor


def train(args):
    print("Getting train data...")

    train_dataset, number_of_channels = prepare_dataset_tiff(args.training_ids_path, args.training_features_path,
                                                             args.S1_band_selection, args.S2_band_selection)

    train_size = int(1 - args.validation_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)

    base_model = select_segmenter(args)

    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    model = Segmenter(model=base_model, learning_rate=args.learning_rate, loss_function=args.loss_function)

    logger = TensorBoardLogger("tb_logs", name=args.model_identifier)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="train/loss",
        mode="min",
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        logger=[logger],
        log_every_n_steps=args.log_step_frequency,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    return model


def load_model(args):
    print("Getting saved model...")

    assert osp.exists(args.current_model_path) is True, "requested model does not exist"

    log_folder_path = args.current_model_path

    version_dir = list(os.scandir(log_folder_path))[args.model_version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")
    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]
    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    base_model = select_segmenter(args)

    ###########################################################

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

    model = Segmenter(model=base_model, learning_rate=args.learning_rate, loss_function=args.loss_function)

    checkpoint = torch.load(str(latest_checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    return model


def create_submissions(args):
    model = load_model(args)

    test_data_path = args.testing_features_path

    with open(args.testing_ids_path, newline='') as f:
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

                for s1_index, band_selection_indicator in enumerate(args.S1_band_selection):

                    if band_selection_indicator == 1:
                        band = np.load(osp.join(s1_folder_path, f"{s1_index}.npy"), allow_pickle=True)
                        all_bands.append(band)

                for s2_index, band_selection_indicator in enumerate(args.S2_band_selection):

                    if band_selection_indicator == 1:
                        band = np.load(osp.join(s2_folder_path, f"{s2_index}.npy"), allow_pickle=True)
                        all_bands.append(band)

                input_tensor = create_tensor(all_bands)

                pred = model(input_tensor.unsqueeze(0))

                pred = pred.cpu().squeeze().detach().numpy()

                all_months.append(pred)

        count = len(all_months)

        if count == 0:
            continue

        agbm_arr = np.asarray(sum(all_months) / count)

        test_agbm_path = osp.join(args.submission_folder_path, f"{id}_agbm.tif")

        im = Image.fromarray(agbm_arr)
        im.save(test_agbm_path)

        if index % 100 == 0:
            print(f"{index} / {total}")


def create_submissions_tiff(args):
    model = load_model(args)

    test_data_path = args.testing_features_path

    with open(args.testing_ids_path, newline='') as f:
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

            S1_data_path = osp.join(test_data_path, f"{id}_S1_{month}.tif")
            S2_data_path = osp.join(test_data_path, f"{id}_S2_{month}.tif")

            bands = []

            if args.S1band_selection.count(1) >= 1:
                S1_bands = rasterio.open(S1_data_path).read().astype(np.float32)
                S1_bands = [x for x, y in zip(S1_bands, args.S1band_selection) if y == 1]
                bands.extend(S1_bands)

            if args.S2band_selection.count(1) >= 1:
                S2_bands = rasterio.open(S2_data_path).read().astype(np.float32)
                S2_bands = [x for x, y in zip(S2_bands, args.S2band_selection) if y == 1]
                bands.extend(S2_bands)

            feature_tensor = create_tensor(bands)

            pred = model(feature_tensor.unsqueeze(0))

            pred = pred.cpu().squeeze().detach().numpy()

            all_months.append(pred)

        count = len(all_months)

        if count == 0:
            continue

        agbm_arr = np.asarray(sum(all_months) / count)

        test_agbm_path = osp.join(args.submission_folder_path, f"{id}_agbm.tif")

        im = Image.fromarray(agbm_arr)
        im.save(test_agbm_path)

        if index % 100 == 0:
            print(f"{index} / {total}")


def set_args():
    model_segmenter = "Unet"
    model_encoder = "efficientnet-b2"
    model_encoder_weights = "imagenet"  # Leave None if not using weights.
    epochs = 5
    learning_rate = 1e-4
    dataloader_workers = 6
    validation_fraction = 0.2
    batch_size = 2
    log_step_frequency = 10
    version = -1  # Keep -1 if loading the latest model version.
    save_top_k_checkpoints = 3
    loss_function = loss_functions.logit_binary_cross_entropy_loss

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

    if model_encoder_weights is not None:
        model_identifier = f"{model_segmenter}_{model_encoder}_{model_encoder_weights}_{s1_bands_indicator}_{s2_bands_indicator}"
    else:
        model_identifier = f"{model_segmenter}_{model_encoder}_{s1_bands_indicator}_{s2_bands_indicator}"

    parser.add_argument('--model_identifier', default=model_identifier, type=str)
    parser.add_argument('--segmenter_name', default=model_segmenter, type=str)
    parser.add_argument('--encoder_name', default=model_encoder, type=str)
    parser.add_argument('--encoder_weights', default=model_encoder_weights, type=str)
    parser.add_argument('--model_version', default=version, type=int)

    data_path = osp.dirname(data.__file__)
    models_path = osp.dirname(models.__file__)
    parser.add_argument('--training_features_path', default=str(osp.join(data_path, "imgs")), type=str)
    parser.add_argument('--training_ids_path', default=str(osp.join(data_path, "patch_names")), type=str)
    parser.add_argument('--testing_features_path', default=str(osp.join(data_path, "testing_converted")), type=str)
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

    parser.add_argument('--S1_band_selection', default=s1_list, type=list)
    parser.add_argument('--S2_band_selection', default=s2_list, type=list)
    parser.add_argument('--loss_function', default=loss_function)

    args = parser.parse_args()

    print('=' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=' * 30)

    return args


if __name__ == '__main__':
    args = set_args()
    train(args)
    create_submissions(args)
