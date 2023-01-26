import torch
from torch.utils.data import Dataset, DataLoader

import os
import rasterio
import warnings
import numpy as np
import os.path as osp

import data
import models
import csv
from models.utils import loss_functions
import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from tqdm import tqdm
import pandas as pd
from torch import nn
from torchgeo.transforms import indices
import models.utils.transforms as tf

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class ChainedSegmentationDatasetMaker(Dataset):
    def __init__(
        self,
        training_feature_path,
        training_labels_path,
        id_list,
        data_type,
        S1_bands,
        S2_bands,
        month_selection,
        transform=None,
        test=False,
    ):
        self.training_feature_path = training_feature_path
        self.training_labels_path = training_labels_path
        self.id_list = id_list
        self.data_type = data_type
        self.S1_bands = S1_bands
        self.S2_bands = S2_bands
        self.transform = transform
        self.month_selection = month_selection
        self.test = test

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):

        id = self.id_list[idx]
        if self.test:
            label_tensor = torch.rand((1, 256, 256))
        else:
            label_path = osp.join(self.training_labels_path, f"{id}_agbm.tif")
            label_tensor = torch.tensor(
                rasterio.open(label_path).read().astype(np.float32)
            )

        tensor_list = []
        for month_index, month_indicator in enumerate(self.month_selection):
            if month_indicator == 0:
                continue
            feature_tensor = retrieve_tiff(
                self.training_feature_path,
                id,
                str(month_index),
                self.S1_bands,
                self.S2_bands,
            )  # .to(torch.float16)
            # print(feature_tensor.shape)
            sample = {"image": feature_tensor, "label": label_tensor}
            feature_tensor = apply_transforms()(sample)
            feature_tensor = feature_tensor["image"]
            tensor_list.append(feature_tensor)

        tensor_list = torch.cat(tensor_list, dim=0)

        return tensor_list, label_tensor


def apply_transforms():
    return nn.Sequential(
        tf.ClampAGBM(
            vmin=0.0, vmax=500.0
        ),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
        indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
        indices.AppendNormalizedDifferenceIndex(
            index_a=11, index_b=12
        ),  # (VV-VH)/(VV+VH), index 16
        indices.AppendNDBI(index_swir=8, index_nir=6),
        # Difference Built-up Index for development detection, index 17
        indices.AppendNDRE(
            index_nir=6, index_vre1=3
        ),  # Red Edge Vegetation Index for canopy detection, index 18
        indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
        indices.AppendNDWI(
            index_green=1, index_nir=6
        ),  # Difference Water Index for water detection, index 20
        indices.AppendSWI(index_vre1=3, index_swir2=8),
    )
    # tf.DropBands(torch.device('cpu'), [0,1,2,3,4,5,6,15,16,18,20]))


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
    with open(args.training_ids_path, newline="") as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    training_features_path = args.tiff_training_features_path

    new_dataset = ChainedSegmentationDatasetMaker(
        training_features_path,
        args.tiff_training_labels_path,
        chip_ids,
        args.data_type,
        args.S1_band_selection,
        args.S2_band_selection,
        args.month_selection,
    )

    return new_dataset


def create_csv_files(args):
    train_dataset = prepare_dataset_training(args)

    train_size = int((1 - args.validation_fraction) * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
    )
    valid_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
    )
    if os.path.exists("../../../../Downloads/train.csv"):
        os.remove("../../../../Downloads/train.csv")
    if os.path.exists("val.csv"):
        os.remove("val.csv")
    with open("../../../../Downloads/train.csv", "a") as f:
        for (x, y) in tqdm(train_dataloader):
            row = np.concatenate(
                [
                    x.detach().cpu().numpy().reshape(x.shape[1], -1).transpose(1, 0),
                    y.detach().cpu().numpy().reshape(y.shape[1], -1).transpose(1, 0),
                ],
                axis=1,
            )
            print(row.shape)
            np.savetxt(f, row)
    with open("val.csv", "a") as f:
        for (x, y) in tqdm(valid_dataloader):
            row = np.concatenate(
                [
                    x.detach().cpu().numpy().reshape(x.shape[1], -1).transpose(1, 0),
                    y.detach().cpu().numpy().reshape(y.shape[1], -1).transpose(1, 0),
                ],
                axis=1,
            )
            print(row.shape)
            np.savetxt(f, row)


def train(args):
    train_dataset = prepare_dataset_training(args)

    train_size = int((1 - args.validation_fraction) * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
    )
    valid_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
    )
    X = []
    Y = []
    c = 0
    lower = 0
    upper = 150
    for (x, y) in tqdm(train_dataloader):
        X.append(x.detach().cpu().numpy().reshape(x.shape[1], -1).transpose(1, 0))
        Y.append(y.detach().cpu().numpy().reshape(y.shape[1], -1).transpose(1, 0))
        c += 1
        # if c < lower:
        #     continue
        # if c > upper:
        #     break
    X = np.array(X)
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    print(X.shape)
    Y = np.array(Y)
    Y = Y.reshape(Y.shape[0] * Y.shape[1], Y.shape[2])
    X = pd.DataFrame(X)
    X["target"] = Y
    subsample_size = 20  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = X.sample(n=subsample_size, random_state=0)
    print(train_data.head())
    metric = "mean_absolute_error"
    save_path = "agModels_2"  # specifies folder to store trained models
    predictor = TabularPredictor(
        label="target", problem_type="regression", path=save_path, eval_metric=metric
    ).fit(
        X,
        time_limit=60 * 60 * 6,
        presets="best_quality",
        holdout_frac=0.05,
        num_cpus=44,
    )
    del X
    del Y

    # predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file
    test_data_nolab = []
    label = []
    c = 0
    lower = 0
    upper = 50
    for (x, y) in tqdm(valid_dataloader):
        test_data_nolab.append(
            x.detach().cpu().numpy().reshape(x.shape[1], -1).transpose(1, 0)
        )
        label.append(y.detach().cpu().numpy().reshape(y.shape[1], -1).transpose(1, 0))
        c += 1
        # if c < lower:
        #     continue
        # if c > upper:
        #     break
    test_data_nolab = np.array(test_data_nolab)
    test_data_nolab = test_data_nolab.reshape(
        test_data_nolab.shape[0] * test_data_nolab.shape[1], test_data_nolab.shape[2]
    )
    label = np.array(label)
    label = label.reshape(label.shape[0] * label.shape[1], label.shape[2])
    test_data_nolab = pd.DataFrame(test_data_nolab)
    test_data = test_data_nolab.copy()
    test_data["target"] = label
    y_pred = predictor.predict(test_data_nolab)
    print("Predictions:  \n", y_pred)
    perf = predictor.evaluate_predictions(
        y_true=test_data["target"], y_pred=y_pred, auxiliary_metrics=True
    )
    predictor.leaderboard(test_data, silent=True)


def create_tensor_from_bands_list(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)
    band_tensor = torch.tensor(band_array)
    # # normalization happens here
    # band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (
    #     band_tensor.std(dim=(1, 2)) + 0.01
    # )
    # band_tensor = band_tensor.permute(2, 0, 1)
    return band_tensor


def set_args():
    band_segmenter = "Unet"
    band_encoder = "efficientnet-b2"
    band_encoder_weights = "imagenet"

    month_segmenter = "Unet"
    month_encoder = "efficientnet-b2"
    month_encoder_weights = "imagenet"

    data_type = "tiff"  # options are "npy" or "tiff"
    epochs = 20
    learning_rate = 1e-4
    dataloader_workers = 12
    validation_fraction = 0.2
    batch_size = 1
    log_step_frequency = 10
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
        "August": 1,
    }

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

    data_path = osp.dirname(data.__file__)
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

    args = parser.parse_args()

    print("=" * 30)
    for arg in vars(args):
        print("--", arg, ":", getattr(args, arg))
    print("=" * 30)

    return args


if __name__ == "__main__":
    args = set_args()
    train(args)
