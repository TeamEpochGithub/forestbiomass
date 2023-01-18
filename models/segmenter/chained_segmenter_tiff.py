import itertools
import operator
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


class ChainedSegmentationDatasetMaker(Dataset):
    def __init__(self, training_feature_path, training_labels_path, id_list, data_type, S1_bands, S2_bands,
                 month_selection, transform=None):
        self.training_feature_path = training_feature_path
        self.training_labels_path = training_labels_path
        self.id_list = id_list
        self.data_type = data_type
        self.S1_bands = S1_bands
        self.S2_bands = S2_bands
        self.transform = transform
        self.month_selection = month_selection

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]

        label_path = osp.join(self.training_labels_path, f"{id}_agbm.tif")
        label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        tensor_list = []

        for month_index, month_indicator in enumerate(self.month_selection):
            feature_tensor = retrieve_tiff(self.training_feature_path, id, str(month_index), self.S1_bands,
                                           self.S2_bands)

            tensor_list.append(feature_tensor)

        tensor_list = torch.cat(tensor_list, dim=0)

        return tensor_list, label_tensor


class ChainedSegmentationSubmissionDatasetMaker(Dataset):
    def __init__(self, testing_feature_path, id_list, data_type, S1_bands, S2_bands, month_selection, transform=None):
        self.testing_feature_path = testing_feature_path
        self.id_list = id_list
        self.data_type = data_type
        self.S1_bands = S1_bands
        self.S2_bands = S2_bands
        self.transform = transform
        self.month_selection = month_selection

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]

        tensor_list = []

        for month_index, month_indicator in enumerate(self.month_selection):
            feature_tensor = retrieve_tiff(self.testing_feature_path, id, str(month_index), self.S1_bands,
                                           self.S2_bands)
            tensor_list.append(feature_tensor)

        return tensor_list


def retrieve_tiff(feature_path, id, month, S1_band_selection, S2_band_selection):
    if int(month) < 10:
        month = "0" + month

    channel_count = S1_band_selection.count(1) + S2_band_selection.count(1)

    S1_path = osp.join(feature_path, f"{id}_S1_{month}.tif")
    S2_path = osp.join(feature_path, f"{id}_S2_{month}.tif")

    if S2_band_selection.count(1) >= 1:
        if not osp.exists(S2_path):
            return create_tensor_from_bands_list(np.zeros((channel_count, 256, 256), dtype=np.float32))

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


class ChainedSegmenter(pl.LightningModule):
    def __init__(self, band_model, month_model, learning_rate, loss_function, repair_mode):
        super().__init__()
        self.band_model = band_model
        self.month_model = month_model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.repair_mode = repair_mode

    def training_step(self, batch, batch_idx):
        x, y = batch

        segmented_bands_list = []
        for index, current_band in enumerate(torch.tensor_split(x, 12, dim=1)):

            if torch.sum(current_band) == 0:
                batch_count = current_band.size(dim=0)
                segmented_bands_list.append(torch.cuda.FloatTensor(batch_count, 1, 256, 256).fill_(0))
                continue

            result = self.band_model(current_band)
            segmented_bands_list.append(result)

        month_tensor = torch.cat(segmented_bands_list, dim=1)

        y_hat = self.month_model(month_tensor)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss) # , on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        segmented_bands_list = []
        for index, current_band in enumerate(torch.tensor_split(x, 12, dim=1)):

            if torch.sum(current_band) == 0:
                batch_count = current_band.size(dim=0)
                segmented_bands_list.append(torch.cuda.FloatTensor(batch_count, 1, 256, 256).fill_(0))
                continue

            result = self.band_model(current_band)
            segmented_bands_list.append(result)

        month_tensor = torch.cat(segmented_bands_list, dim=1)

        y_hat = self.month_model(month_tensor)
        loss = self.loss_function(y_hat, y)
        self.log("val/loss", loss) #, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):

        segmented_bands_list = []
        for index, current_band in enumerate(torch.tensor_split(x, 12, dim=1)):

            if torch.sum(current_band) == 0:
                batch_count = current_band.size(dim=0)
                segmented_bands_list.append(torch.cuda.FloatTensor(batch_count, 1, 256, 256).fill_(0))
                continue

            result = self.band_model(current_band)
            segmented_bands_list.append(result)

        month_tensor = torch.cat(segmented_bands_list, dim=1)

        y_hat = self.month_model(month_tensor)

        return y_hat


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


def prepare_dataset_testing(args):
    with open(args.testing_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    testing_features_path = args.tiff_testing_features_path

    new_dataset = ChainedSegmentationSubmissionDatasetMaker(testing_features_path,
                                                            chip_ids,
                                                            args.data_type,
                                                            args.S1_band_selection,
                                                            args.S2_band_selection,
                                                            args.month_selection)
    return new_dataset, chip_ids


def select_segmenter(segmenter_name, encoder_name, encoder_weights, channel_count):
    if segmenter_name == "Unet":

        base_model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
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

    train_dataset = prepare_dataset_training(args)

    train_size = int((1 - args.validation_fraction) * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)

    band_channel_count = (args.S1_band_selection.count(1) + args.S2_band_selection.count(1))
    month_channel_count = args.month_selection.count(1)

    band_segmenter_model = select_segmenter(args.band_segmenter_name,
                                            args.band_encoder_name,
                                            args.band_encoder_weights_name,
                                            band_channel_count)

    month_segmenter_model = select_segmenter(args.month_segmenter_name,
                                             args.month_encoder_name,
                                             args.month_encoder_weights_name,
                                             month_channel_count)

    model = ChainedSegmenter(band_model=band_segmenter_model,
                             month_model=month_segmenter_model,
                             learning_rate=args.learning_rate,
                             loss_function=args.loss_function,
                             repair_mode=args.missing_month_repair_mode)

    logger = TensorBoardLogger("tb_logs", name=args.model_identifier)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="val/loss",
        mode="min",
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        strategy='dp',
        max_epochs=args.epochs,
        logger=[logger],
        log_every_n_steps=args.log_step_frequency,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0
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

    band_channel_count = (args.S1_band_selection.count(1) + args.S2_band_selection.count(1))
    month_channel_count = args.month_selection.count(1)

    band_segmenter_model = select_segmenter(args.band_segmenter_name,
                                            args.band_encoder_name,
                                            args.band_encoder_weights_name,
                                            band_channel_count)

    month_segmenter_model = select_segmenter(args.month_segmenter_name,
                                             args.month_encoder_name,
                                             args.month_encoder_weights_name,
                                             month_channel_count)

    model = ChainedSegmenter(band_model=band_segmenter_model,
                             month_model=month_segmenter_model,
                             learning_rate=args.learning_rate,
                             loss_function=args.loss_function,
                             repair_mode=args.missing_month_repair_mode)

    checkpoint = torch.load(str(latest_checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    return model


def create_submissions(args):
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

    new_dataset, chip_ids = prepare_dataset_testing(args)

    trainer = Trainer(accelerator="gpu", devices=1)

    dl = DataLoader(new_dataset, num_workers=12)

    predictions = trainer.predict(model, dataloaders=dl)

    transformed_predictions = [x.cpu().squeeze().detach().numpy() for x in predictions]

    linked_tensor_list = list(zip(chip_ids, transformed_predictions))

    linked_tensor_list = sorted(linked_tensor_list, key=operator.itemgetter(0))

    averaged_tensor_list = list(accumulate_predictions(linked_tensor_list))

    for id_tensor_pair in averaged_tensor_list:
        current_id = id_tensor_pair[0]
        current_tensor = id_tensor_pair[1]

        agbm_path = osp.join(args.submission_folder_path, f"{current_id}_agbm.tif")

        im = Image.fromarray(current_tensor)
        im.save(agbm_path)

    print("Finished creating submission.")


def chained_experimental_submission(args):
    model = load_model(args)

    new_dataset, chip_ids = prepare_dataset_testing(args)

    trainer = Trainer(accelerator="gpu", devices=1)

    dl = DataLoader(new_dataset, num_workers=12)

    predictions = trainer.predict(model, dataloaders=dl)

    transformed_predictions = [x.cpu().squeeze().detach().numpy() for x in predictions]

    linked_tensor_list = list(zip(chip_ids, transformed_predictions))

    for id_tensor_pair in linked_tensor_list:
        current_id = id_tensor_pair[0]
        current_tensor = id_tensor_pair[1]

        agbm_path = osp.join(args.submission_folder_path, f"{current_id}_agbm.tif")

        im = Image.fromarray(current_tensor)
        im.save(agbm_path)

    print("Finished creating submission.")


def set_args():
    band_segmenter = "Unet"
    band_encoder = "efficientnet-b2"
    band_encoder_weights = "imagenet"

    month_segmenter = "Unet"
    month_encoder = "efficientnet-b2"
    month_encoder_weights = "imagenet"

    data_type = "tiff"  # options are "npy" or "tiff"
    epochs = 40
    learning_rate = 1e-5
    dataloader_workers = 32
    validation_fraction = 0.2
    batch_size = 64
    log_step_frequency = 200
    version = -1  # Keep -1 if loading the latest model version.
    save_top_k_checkpoints = 3
    loss_function = loss_functions.rmse_loss

    missing_month_repair_mode = "zeros"

    month_selection = {
        "September": 1,
        "October": 1,
        "November": 1,
        "December": 1,
        "January": 1,
        "February": 1,
        "March": 1,
        "April": 1,
        "May": 1,
        "June": 1,
        "July": 1,
        "August": 1
    }

    month_list = list(month_selection.values())

    month_selection_indicator = "months-" + ''.join(str(x) for x in month_list)

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

    if band_encoder_weights is not None and month_encoder_weights is not None:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{band_encoder_weights}_{s1_bands_indicator}_{s2_bands_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"

    elif band_encoder_weights is None and month_encoder_weights is not None:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{s1_bands_indicator}_{s2_bands_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"

    elif band_encoder_weights is not None and month_encoder_weights is None:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{s1_bands_indicator}_{s2_bands_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"
    else:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{s1_bands_indicator}_{s2_bands_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_selection_indicator}"

    parser.add_argument('--model_identifier', default=model_identifier, type=str)

    parser.add_argument('--band_segmenter_name', default=band_segmenter, type=str)
    parser.add_argument('--band_encoder_name', default=band_encoder, type=str)
    parser.add_argument('--band_encoder_weights_name', default=band_encoder_weights, type=str)

    parser.add_argument('--month_segmenter_name', default=month_segmenter, type=str)
    parser.add_argument('--month_encoder_name', default=month_encoder, type=str)
    parser.add_argument('--month_encoder_weights_name', default=month_encoder_weights, type=str)

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

    parser.add_argument('--S1_band_selection', default=s1_list, type=list)
    parser.add_argument('--S2_band_selection', default=s2_list, type=list)
    parser.add_argument('--month_selection', default=month_list, type=list)
    parser.add_argument('--loss_function', default=loss_function)

    parser.add_argument('--missing_month_repair_mode', default=missing_month_repair_mode, type=str)

    args = parser.parse_args()

    print('=' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=' * 30)

    return args


if __name__ == '__main__':
    args = set_args()
    train(args)
    # create_submissions(args)
    # chained_experimental_submission(args)
