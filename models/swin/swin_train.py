import itertools
import operator
import sys

import torch
from PIL import Image
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
import models
import data
import csv
import torch.nn.functional as F
import argparse
from models.res_swin import Res_Swin_v1
from models.utils.warmup_scheduler.scheduler import GradualWarmupScheduler
import random
import copy

random.seed(0)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SegmentationDatasetMakerConverted(Dataset):
    def __init__(self, training_feature_path, chip_ids, s1_bands, s2_bands, transform=None):
        self.training_feature_path = training_feature_path
        self.chip_ids = chip_ids
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.transform = transform

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]

        label_path = osp.join(self.training_feature_path, chip_id, "label.npy")
        label = np.load(label_path, allow_pickle=True)
        label_tensor = torch.tensor(np.asarray([label], dtype=np.float32))

        arr_list = []
        for month in range(12):
            if month in [1,2,3,4]:
                continue
            for index, s1_index in enumerate(self.s1_bands):
                if s1_index==1:
                    if osp.exists(osp.join(self.training_feature_path, chip_id, str(month), "S1", f"{index}.npy")):
                        band = np.load(osp.join(self.training_feature_path, chip_id, str(month), "S1", f"{index}.npy"),allow_pickle=True)
                        arr_list.append(band)
            for index, s2_index in enumerate(self.s2_bands):
                if s2_index == 1:
                    if osp.exists(osp.join(self.training_feature_path, chip_id, str(month), "S2", f"{index}.npy")):
                        band = np.load(osp.join(self.training_feature_path, chip_id, str(month), "S2", f"{index}.npy"),allow_pickle=True)
                        arr_list.append(band)
        if len(arr_list)<112:
            for i in range(112-len(arr_list)):
                random.shuffle(arr_list)
                arr_list.append(copy.deepcopy(arr_list[0]))
        data_tensor = create_tensor_from_bands_list(arr_list)
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor


class SegmentationDatasetMakerTIFF(Dataset):
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
            month = "0"+month

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

        feature_tensor = create_tensor_from_bands_list(bands)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
        feature_tensor = torch.mean(feature_tensor, dim=0, keepdim=True)
        return feature_tensor, label_tensor


class SubmissionDatasetMakerConverted(Dataset):
    def __init__(self, training_data_path, chip_ids, s1_bands, s2_bands, transform=None):
        self.training_data_path = training_data_path
        self.chip_ids = chip_ids
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.transform = transform

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):

        chip_id = self.chip_ids[idx]
        arr_list = []
        for month in range(12):
            if month in [1, 2, 3, 4]:
                continue
            for index, s1_index in enumerate(self.s1_bands):
                if s1_index==1:
                    if osp.exists(osp.join(self.training_data_path, chip_id, str(month), "S1", f"{index}.npy")):
                        band = np.load(osp.join(self.training_data_path, chip_id, str(month), "S1", f"{index}.npy"),allow_pickle=True)
                        arr_list.append(band)
            for index, s2_index in enumerate(self.s2_bands):
                if s2_index == 1:
                    if osp.exists(osp.join(self.training_data_path, chip_id, str(month), "S2", f"{index}.npy")):
                        band = np.load(osp.join(self.training_data_path, chip_id, str(month), "S2", f"{index}.npy"),allow_pickle=True)
                        arr_list.append(band)
        if len(arr_list)<112:
            for i in range(112-len(arr_list)):
                random.shuffle(arr_list)
                arr_list.append(copy.deepcopy(arr_list[0]))
        data_tensor = create_tensor_from_bands_list(arr_list)
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor


class SubmissionDatasetMakerTiff(Dataset):
    def __init__(self, training_feature_path, id_month_list, S1_bands, S2_bands, transform=None):
        self.training_feature_path = training_feature_path
        self.id_month_list = id_month_list
        self.S1_bands = S1_bands
        self.S2_bands = S2_bands
        self.transform = transform

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):

        id, month = self.id_month_list[idx]

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

        feature_tensor = create_tensor_from_bands_list(bands)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)

        return feature_tensor


class Segmenter(pl.LightningModule):
    def __init__(self, model, epochs, warmup_epochs, learning_rate, weight_decay, loss_function):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss)
        self.scheduler.step()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs - self.warmup_epochs,
                                                                eta_min=1e-6)
        self.scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=1, total_epoch=self.warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        return [optimizer], [self.scheduler]

    def forward(self, x):
        return self.model(x)


def prepare_dataset_converted(training_ids_path, training_features_path, S1_band_selection, S2_band_selection):


    with open(training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    new_dataset = SegmentationDatasetMakerConverted(training_features_path, chip_ids, S1_band_selection,
                                                    S2_band_selection)
    return new_dataset, (S1_band_selection.count(1) + S2_band_selection.count(1))


def prepare_dataset_tiff(training_ids_path, training_features_path, training_labels_path, S1_band_selection, S2_band_selection):
    with open(training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    id_month_list = []
    for id in chip_ids:
        for month in range(0, 12):

            month_patch_path = osp.join(training_features_path, f"{id}_S2_{month:02}.tif")
            if osp.exists(month_patch_path):
                id_month_list.append((id, str(month)))

    new_dataset = SegmentationDatasetMakerTIFF(training_features_path, training_labels_path, id_month_list, S1_band_selection,
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


def create_tensor_from_bands_list(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)

    band_tensor = torch.tensor(band_array)

    # normalization happens here
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (band_tensor.std(dim=(1, 2)) + 0.01)
    band_tensor = band_tensor.permute(2, 0, 1)

    return band_tensor


def train(args):

    print("Getting train data...")

    if args.data_type == "npy":
        train_dataset, number_of_channels = prepare_dataset_converted(args.training_ids_path,
                                                                      args.converted_training_features_path,
                                                                      args.S1_band_selection,
                                                                      args.S2_band_selection)

    elif args.data_type == "tiff":
        train_dataset, number_of_channels = prepare_dataset_tiff(args.training_ids_path,
                                                                 args.tiff_training_features_path,
                                                                 args.tiff_training_labels_path,
                                                                 args.S1_band_selection,
                                                                 args.S2_band_selection)
    else:
        sys.exit("Invalid data type selected during training.")

    train_size = int((1 - args.validation_fraction) * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_workers)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_workers)

    base_model = Res_Swin_v1()

    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.load_state_dict(torch.load(pre_trained_weights_path))

    model = Segmenter(model=base_model, epochs=args.epochs, warmup_epochs=args.warmup_epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay, loss_function=args.val_loss_function)

    logger = TensorBoardLogger("tb_logs", name=args.model_identifier)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="val/loss",
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

    print("Getting saved corrupted_model...")

    assert osp.exists(args.current_model_path) is True, "requested corrupted_model does not exist"

    log_folder_path = args.current_model_path

    version_dir = list(os.scandir(log_folder_path))[args.model_version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")
    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]
    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    base_model = Res_Swin_v1()

    ###########################################################

    # This block might be redundant if we can download weights via the python segmentation models library.
    # However, it might be that not all weights are available this way.
    # If you have downloaded weights (in the .pt format), put them in the pre-trained-weights folder
    # and give the file the same name as the encoder you're using.
    # If you do that, this block will try and load them for your corrupted_model.
    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.load_state_dict(torch.load(pre_trained_weights_path))

    ###########################################################

    model = Segmenter(model=base_model, learning_rate=args.learning_rate, loss_function=args.val_loss_function)

    checkpoint = torch.load(str(latest_checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    return model


def create_submissions(args):
    if args.data_type == "npy":
        create_submissions_converted(args)
    elif args.data_type == "tiff":
        create_submissions_tiff(args)
    else:
        sys.exit("Invalid data type selected during submission creation.")


def create_submissions_converted(args):

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

                input_tensor = create_tensor_from_bands_list(all_bands)

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

    test_data_path = args.tiff_testing_features_path

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

            if args.S1_band_selection.count(1) >= 1:
                S1_bands = rasterio.open(S1_data_path).read().astype(np.float32)
                S1_bands = [x for x, y in zip(S1_bands, args.S1_band_selection) if y == 1]
                bands.extend(S1_bands)

            if args.S2_band_selection.count(1) >= 1:
                S2_bands = rasterio.open(S2_data_path).read().astype(np.float32)
                S2_bands = [x for x, y in zip(S2_bands, args.S2_band_selection) if y == 1]
                bands.extend(S2_bands)

            if len(bands) == 0:
                continue

            feature_tensor = create_tensor_from_bands_list(bands)

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


# Source: https://stackoverflow.com/a/2249060/14633351
def accumulate_predictions(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        group_list = list(subiter)
        total = sum(tensors for tensor_id, tensors in group_list)
        yield key, total / len(group_list)


def experimental_submission(args):

    model = load_model(args)

    with open(args.testing_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    id_month_list = []

    if args.data_type == "npy":

        # for id in chip_ids:
        #     for month in range(0, 12):
        #         month_patch_path = osp.join(args.converted_testing_features_path, id, str(month))
        #         if osp.exists(osp.join(month_patch_path, "S2")):
        #             id_month_list.append((id, str(month)))

        new_dataset = SubmissionDatasetMakerConverted(args.converted_testing_features_path,
                                                      chip_ids,
                                                      args.S1_band_selection,
                                                      args.S2_band_selection)

    elif args.data_type == "tiff":

        for id in chip_ids:
            for month in range(0, 12):

                month_patch_path = osp.join(args.tiff_testing_features_path, f"{id}_S2_{month:02}.tif")
                if osp.exists(month_patch_path):
                    id_month_list.append((id, str(month)))

        new_dataset = SubmissionDatasetMakerTiff(args.tiff_testing_features_path,
                                                 id_month_list,
                                                 args.S1_band_selection,
                                                 args.S2_band_selection)

    else:
        sys.exit("Error: Invalid data type selected.")

    trainer = Trainer(accelerator="gpu", devices=1)

    dl = DataLoader(new_dataset, num_workers=20)

    predictions = trainer.predict(model, dataloaders=dl)

    transformed_predictions = [x.cpu().squeeze().detach().numpy() for x in predictions]

    tensor_id_list = [i[0] for i in id_month_list]

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
    data_type = "npy"  # options are "npy" or "tiff"
    epochs = 30
    warmup_epochs = 10
    learning_rate = 1e-4
    weight_decay = 5e-5
    dataloader_workers = 6
    validation_fraction = 0.2
    batch_size = 64
    log_step_frequency = 25
    version = -1  # Keep -1 if loading the latest corrupted_model version.
    save_top_k_checkpoints = 3
    loss_function = F.mse_loss

    sentinel_1_bands = {
        "VV ascending": 1,
        "VH ascending": 1,
        "VV descending": 1,
        "VH descending": 1
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


    model_identifier = f"res_swin_v1_{s1_bands_indicator}_{s2_bands_indicator}"
    parser.add_argument('--model_identifier', default=model_identifier, type=str)
    parser.add_argument('--encoder_name', default="res_swin_v1", type=str)
    parser.add_argument('--model_version', default=version, type=int)
    parser.add_argument('--data_type', default=data_type, type=str)

    data_path = osp.dirname(data.__file__)
    models_path = osp.dirname(models.__file__)

    data_path = r"C:\Users\Team Epoch A\Documents\Epoch III\forestbiomass\data"

    # Note: Converted data does not have an explicit label path, as labels are stored within training_features
    parser.add_argument('--converted_training_features_path', default=str(osp.join(data_path, "converted")), type=str)
    parser.add_argument('--converted_testing_features_path', default=str(osp.join(data_path, "testing_converted")), type=str)

    parser.add_argument('--tiff_training_features_path', default=str(osp.join(data_path, "imgs", "train_features")))
    parser.add_argument('--tiff_training_labels_path', default=str(osp.join(data_path, "imgs", "train_agbm")))
    parser.add_argument('--tiff_testing_features_path', default=str(osp.join(data_path, "imgs", "test_features")))

    parser.add_argument('--training_ids_path', default=str(osp.join(data_path, "patch_names")), type=str)
    parser.add_argument('--testing_ids_path', default=str(osp.join(data_path, "test_patch_names")), type=str)

    parser.add_argument('--current_model_path', default=str(osp.join(models_path, "tb_logs", model_identifier)), type=str)
    parser.add_argument('--submission_folder_path', default=str(osp.join(data_path, "imgs", "test_agbm")), type=str)

    parser.add_argument('--dataloader_workers', default=dataloader_workers, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--warmup_epochs', default=warmup_epochs, type=int)
    parser.add_argument('--learning_rate', default=learning_rate, type=float)
    parser.add_argument('--weight_decay', default=weight_decay, type=float)
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
    # #create_submissions(args)
    # experimental_submission(args)

    # directory = os.fsencode(args.tiff_testing_features_path)
    # current_filename = "XXXXXXXX"
    #
    # all_data = {}
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     # print(len(filename))
    #     if len(filename) > 18:
    #         print(filename)


