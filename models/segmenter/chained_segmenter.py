import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:23900'

import itertools
import operator
import sys

import torch
from PIL import Image
import numpy
from torch.utils.data import Dataset, DataLoader
from torch import distributed as dist
import segmentation_models_pytorch as smp

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
import torch.nn as nn
from models.se_net import SqEx
from torch import nn
from torchgeo.transforms import indices
import models.utils.transforms as tf
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class ChainedSegmentationDatasetMaker(Dataset):
    def __init__(self, training_feature_path, training_labels_path, id_list, data_type,
                 band_selection, month_selection, data_augmentation_pipeline, transform=None):
        self.training_feature_path = training_feature_path
        self.training_labels_path = training_labels_path
        self.id_list = id_list
        self.data_type = data_type
        self.transform = transform
        self.band_selection = band_selection
        self.month_selection = month_selection
        self.data_augmentation_pipeline = data_augmentation_pipeline

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):

        id = self.id_list[idx]

        label_path = osp.join(self.training_labels_path, f"{id}_agbm.tif")
        label_image = rasterio.open(label_path).read().astype(np.float32)

        all_bands = []

        for month_index, month_indicator in enumerate(self.month_selection):
            band_images = retrieve_tiff(self.training_feature_path, id, str(month_index), self.band_selection)

            all_bands.extend(band_images)

        transformed_images_dict = albumentation_input_wrapper(all_bands, label_image, self.data_augmentation_pipeline)

        label_tensor = torch.from_numpy(np.asarray(transformed_images_dict['mask'], dtype=np.float32).copy())

        tensor_list = []

        for index, current_band in enumerate(
                np.array_split(np.asarray(transformed_images_dict['image']), len(self.month_selection))):
            normalized_tensors = torch.tensor(current_band)

            dictionary_tensor = {'image': normalized_tensors}

            expanded_tensor = select_bands(bands_to_keep=self.band_selection)(dictionary_tensor)['image']

            tensor_list.append(expanded_tensor)

        tensor_list = torch.cat(tensor_list, dim=0)

        return tensor_list, label_tensor


def albumentation_input_wrapper(images, label_image, augmenter):

    return augmenter(image=np.asarray(images), mask=label_image)


class ChainedSegmentationSubmissionDatasetMaker(Dataset):
    def __init__(self, testing_feature_path, id_list, data_type, band_selection, month_selection, transform=None):
        self.testing_feature_path = testing_feature_path
        self.id_list = id_list
        self.data_type = data_type
        self.transform = transform
        self.band_selection = band_selection
        self.month_selection = month_selection

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]

        tensor_list = []

        for month_index, month_indicator in enumerate(self.month_selection):
            base_tensor = retrieve_tiff(self.testing_feature_path, id, str(month_index), self.band_selection)

            dictionary_tensor = {'image': base_tensor}  # Dict required for usage of torch transforms

            expanded_tensor = select_bands(bands_to_keep=self.band_selection)(dictionary_tensor)

            tensor_list.append(expanded_tensor)

        tensor_list = torch.cat(tensor_list, dim=0)

        return tensor_list


def retrieve_tiff(feature_path, id, month, band_selection):
    if int(month) < 10:
        month = "0" + month

    channel_count = 15  # Since bands are removed later, we need to ensure 15 bands are always returned here.

    S1_path = osp.join(feature_path, f"{id}_S1_{month}.tif")
    S2_path = osp.join(feature_path, f"{id}_S2_{month}.tif")

    if not osp.exists(S2_path):
        return np.zeros((channel_count, 256, 256), dtype=np.float32)

    bands = []

    S1_bands = rasterio.open(S1_path).read().astype(np.float32)
    bands.extend(S1_bands)

    S2_bands = rasterio.open(S2_path).read().astype(np.float32)
    bands.extend(S2_bands)

    # feature_tensor = create_tensor_from_bands_list(bands)

    return bands


def select_bands(bands_to_keep):
    return nn.Sequential(
        indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
        indices.AppendNormalizedDifferenceIndex(index_a=11, index_b=12),  # (VV-VH)/(VV+VH), index 16
        indices.AppendNDBI(index_swir=8, index_nir=6),
        # Difference Built-up Index for development detection, index 17
        indices.AppendNDRE(index_nir=6, index_vre1=3),  # Red Edge Vegetation Index for canopy detection, index 18
        indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
        indices.AppendNDWI(index_green=1, index_nir=6),  # Difference Water Index for water detection, index 20
        indices.AppendSWI(index_vre1=3, index_swir2=8),
        # Standardized Water-Level Index for water detection, index 21
        tf.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
        tf.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
        tf.DropBands(torch.device('cpu'), bands_to_keep),  # DROPS ALL BUT SPECIFIED bands_to_keep
    )


class ChainedSegmenter(pl.LightningModule):
    def __init__(self, band_model, month_model, learning_rate, loss_function, repair_mode, band_count, month_count):
        super().__init__()
        self.band_model = band_model
        self.month_model = month_model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.repair_mode = repair_mode
        self.band_count = band_count
        self.month_count = month_count

    def training_step(self, batch, batch_idx):
        x, y = batch

        normalizer = nn.BatchNorm2d(self.band_count).cuda()

        segmented_bands_list = []
        for index, current_band in enumerate(torch.tensor_split(x, self.month_count, dim=1)):

            if torch.sum(current_band) == 0:
                batch_count = current_band.size(dim=0)
                segmented_bands_list.append(torch.cuda.FloatTensor(batch_count, 1, 256, 256).fill_(0))
                continue

            result = self.band_model(normalizer(current_band))

            segmented_bands_list.append(result)

        month_tensor = torch.cat(segmented_bands_list, dim=1)

        y_hat = self.month_model(month_tensor)
        loss = self.loss_function(y_hat, y)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        normalizer = nn.BatchNorm2d(self.band_count).cuda()

        segmented_bands_list = []
        for index, current_band in enumerate(torch.tensor_split(x, self.month_count, dim=1)):

            if torch.sum(current_band) == 0:
                batch_count = current_band.size(dim=0)
                segmented_bands_list.append(torch.cuda.FloatTensor(batch_count, 1, 256, 256).fill_(0))
                continue

            result = self.band_model(normalizer(current_band).to('cuda'))
            segmented_bands_list.append(result)

        month_tensor = torch.cat(segmented_bands_list, dim=1)

        y_hat = self.month_model(month_tensor)
        loss = self.loss_function(y_hat, y)
        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        normalizer = nn.BatchNorm2d(self.band_count).cuda()

        segmented_bands_list = []
        for index, current_band in enumerate(torch.tensor_split(x, self.month_count, dim=1)):

            if torch.sum(current_band) == 0:
                batch_count = current_band.size(dim=0)
                segmented_bands_list.append(torch.cuda.FloatTensor(batch_count, 1, 256, 256).fill_(0))
                continue

            result = self.band_model(normalizer(current_band).to('cuda'))
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
                                                  args.band_selection,
                                                  args.month_selection,
                                                  args.data_augmentation_pipeline)

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
                                                            args.band_selection,
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
    elif segmenter_name == "Unet++":

        base_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "MAnet":

        base_model = smp.MAnet(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "Linknet":

        base_model = smp.Linknet(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "FPN":

        base_model = smp.FPN(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "PSPNet":

        base_model = smp.PSPNet(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "PAN":

        base_model = smp.PAN(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "DeepLabV3":

        base_model = smp.DeepLabV3(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    elif segmenter_name == "DeepLabV3+":

        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            in_channels=channel_count,
            classes=1,
            encoder_weights=encoder_weights
        )
    else:
        base_model = None

    assert base_model is not None, "Segmenter name was not recognized."
    return base_model


def create_tensor_from_bands_list(band_array):
    # band_array = np.asarray(band_list, dtype=np.float32)

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

    band_channel_count = len(args.band_selection)
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
                             repair_mode=args.missing_month_repair_mode,
                             band_count=len(args.band_selection),
                             month_count=args.month_selection.count(1))

    logger = TensorBoardLogger("../tb_logs", name=args.model_identifier)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="val/loss",
        mode="min",
    )

    strategy = args.multiprocessing_strategy

    if strategy == "ddp":
        strategy = DDPStrategy(process_group_backend="gloo")

    trainer = Trainer(
        accelerator="gpu",
        devices=args.device_count,
        max_epochs=args.epochs,
        logger=[logger],
        log_every_n_steps=args.log_step_frequency,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        strategy=strategy
    )

    if args.warm_start:

        assert osp.exists(args.current_model_path) is True, "requested model does not exist"

        log_folder_path = args.current_model_path

        version_dir = list(os.scandir(log_folder_path))[args.model_version]

        checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")
        latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]
        latest_checkpoint_path = str(osp.join(checkpoint_dir_path, latest_checkpoint_name))

        print("Starting WARM start.")

        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader,
                    ckpt_path=latest_checkpoint_path)
    else:
        print("Starting COLD start.")
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=valid_dataloader)

    return model


def load_model(args):
    print("Getting saved model...")

    assert osp.exists(args.current_model_path) is True, "requested model does not exist"

    log_folder_path = args.current_model_path

    version_dir = list(os.scandir(log_folder_path))[args.model_version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")
    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]
    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    band_channel_count = len(args.band_selection)
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
                             repair_mode=args.missing_month_repair_mode,
                             band_count=len(args.band_selection),
                             month_count=args.month_selection.count(1))

    checkpoint = torch.load(str(latest_checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    return model


# Source: https://stackoverflow.com/a/2249060/14633351
def accumulate_predictions(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        group_list = list(subiter)
        total = sum(tensors for tensor_id, tensors in group_list)
        yield key, total / len(group_list)


def submission_generator(args):
    model = load_model(args)

    new_dataset, chip_ids = prepare_dataset_testing(args)

    trainer = Trainer(accelerator="cpu", devices=1)

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
    band_encoder = "efficientnet-b0"
    band_encoder_weights = "imagenet"

    month_segmenter = "Unet++"
    month_encoder = "efficientnet-b0"
    month_encoder_weights = "imagenet"

    data_type = "tiff"  # options are "npy" or "tiff"
    epochs = 100
    learning_rate = 1e-4
    dataloader_workers = 12
    validation_fraction = 0.2
    batch_size = 8
    log_step_frequency = 50
    version = -1  # Keep -1 if loading the latest model version.
    save_top_k_checkpoints = 1
    loss_function = loss_functions.rmse_loss

    missing_month_repair_mode = "zeros"

    multiprocessing_strategy = "ddp"  # replace with ddp if using more than 1 device
    device_count = 2

    warm_start = True

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

    band_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

    band_indicator = ["1" if k in band_selection else "0" for k, v in band_map.items()]
    band_indicator.insert(11, "-")
    band_indicator.insert(-9, "-")

    band_selection_indicator = "bands-" + ''.join(str(x) for x in band_indicator)

    if band_encoder_weights is not None and month_encoder_weights is not None:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{band_encoder_weights}_{band_selection_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"

    elif band_encoder_weights is None and month_encoder_weights is not None:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{band_selection_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"

    elif band_encoder_weights is not None and month_encoder_weights is None:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{band_selection_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_encoder_weights}_{month_selection_indicator}"
    else:
        model_identifier = f"Bands_{band_segmenter}_{band_encoder}_{band_selection_indicator}" \
                           f"_Months_{month_segmenter}_{month_encoder}_{month_selection_indicator}"

    data_augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    parser = argparse.ArgumentParser()

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

    data_path = r"C:\Users\kuipe\Desktop\Epoch\forestbiomass\data"

    # Note: Converted data does not have an explicit label path, as labels are stored within training_features
    parser.add_argument('--converted_training_features_path', default=str(osp.join(data_path, "converted")), type=str)
    parser.add_argument('--converted_testing_features_path', default=str(osp.join(data_path, "testing_converted")),
                        type=str)

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

    parser.add_argument('--band_selection', default=band_selection, type=list)
    parser.add_argument('--month_selection', default=month_list, type=list)
    parser.add_argument('--loss_function', default=loss_function)

    parser.add_argument('--missing_month_repair_mode', default=missing_month_repair_mode, type=str)

    parser.add_argument('--multiprocessing_strategy', default=multiprocessing_strategy, type=str)
    parser.add_argument('--device_count', default=device_count, type=int)

    parser.add_argument('--data_augmentation_pipeline', default=data_augmentation_pipeline)

    parser.add_argument('--warm_start', default=warm_start, type=str)

    args = parser.parse_args()

    print('=' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=' * 30)

    return args


if __name__ == '__main__':
    args = set_args()
    train(args)
    submission_generator(args)
