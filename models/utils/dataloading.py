import rasterio
from torch import nn
from torchgeo.transforms import indices

import models.utils.transforms as tf
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import torch
import warnings
import random
import copy

random.seed(0)

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SegmentationDatasetMaker(Dataset):
    def __init__(self, args, id_month_list, corrupted_transform_method):
        self.training_data_path = args.converted_training_features_path
        self.bands_to_keep = args.bands_to_keep
        self.id_month_list = id_month_list
        self.corrupted_transform_method = corrupted_transform_method

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):

        id, month = self.id_month_list[idx]

        label_path = osp.join(self.training_data_path, id, "label.npy")
        label = np.load(label_path, allow_pickle=True)
        label_tensor = torch.tensor(np.asarray([label], dtype=np.float32))

        bands = []

        for index in range(11):
            band = np.load(osp.join(self.training_data_path, id, month, "S2", f"{index}.npy"), allow_pickle=True)
            bands.append(band)

        for index in range(4):
            band = np.load(osp.join(self.training_data_path, id, month, "S1", f"{index}.npy"), allow_pickle=True)
            bands.append(band)

        all_tensor = create_tensor(bands)

        sample = {'image': all_tensor, 'label': label_tensor}  # 'image' and 'label' are used by torchgeo
        selected_tensor = apply_transforms(bands_to_keep=self.bands_to_keep,
                                           corrupted_transform_method=self.corrupted_transform_method)(sample)

        return selected_tensor['image'], selected_tensor['label']

class SegmentationDatasetMakerTIFF(Dataset):
    def __init__(self, args, id_month_list, corrupted_transform_method):
        self.training_feature_path = args.tiff_training_features_path
        self.training_label_path = args.tiff_training_labels_path
        self.id_month_list = id_month_list
        self.bands_to_keep = args.bands_to_keep
        self.id_month_list = id_month_list
        self.corrupted_transform_method = corrupted_transform_method

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

        S2_bands = rasterio.open(S2_data_path).read().astype(np.float32)
        bands.extend(S2_bands)

        S1_bands = rasterio.open(S1_data_path).read().astype(np.float32)
        bands.extend(S1_bands)

        all_tensor = create_tensor(bands)

        sample = {'image': all_tensor, 'label': label_tensor}  # 'image' and 'label' are used by torchgeo
        selected_tensor = apply_transforms(bands_to_keep=self.bands_to_keep,
                                           corrupted_transform_method=self.corrupted_transform_method)(sample)

        return selected_tensor['image'], selected_tensor['label']

class SubmissionDataLoader(Dataset):
    def __init__(self, args, id_month_list, corrupted_transform_method):
        self.training_data_path = args.converted_testing_features_path
        self.id_month_list = id_month_list
        self.corrupted_transform_method = corrupted_transform_method
        self.bands_to_keep = args.bands_to_keep

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):
        id, month = self.id_month_list[idx]

        label_tensor = torch.rand((256, 256))

        all_list = []

        for index in range(11):
            band = np.load(osp.join(self.training_data_path, id, month, "S2", f"{index}.npy"), allow_pickle=True)
            all_list.append(band)

        for index in range(4):
            band = np.load(osp.join(self.training_data_path, id, month, "S1", f"{index}.npy"), allow_pickle=True)
            all_list.append(band)

        all_tensor = create_tensor(all_list)

        transforms = nn.Sequential(
            # tf.ClampAGBM(vmin=0., vmax=500.),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
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
            tf.DropBands(torch.device('cpu'), self.bands_to_keep),  # DROPS ALL BUT SPECIFIED bands_to_keep
            self.corrupted_transform_method  # Applies corrupted band transformation
        )

        sample = {'image': all_tensor, 'label': label_tensor}  # 'image' and 'label' are used by torchgeo
        selected_tensor = transforms(sample)
        return selected_tensor['image']


def create_tensor(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)

    band_array[np.where(np.all(band_array != -9999.0, axis=(1, 2)))] = np.zeros_like(
        band_array[np.where(np.all(band_array != -9999.0, axis=(1, 2)))])

    # band_means = np.array([-11.299535371573057,
    #                        -17.81367613586265,
    #                        -1143.9031632757003,
    #                        -1149.7795683922136,
    #                        1475.7613735137484,
    #                        1540.245956459104,
    #                        1533.271091203123,
    #                        1856.3387101594592,
    #                        2456.316603394904,
    #                        2575.698182361417,
    #                        2721.755552212775,
    #                        2669.675785310289,
    #                        976.1367830278934,
    #                        637.262333553942,
    #                        15.068401727931002])
    #
    # band_stds = np.array([2.3706742546317616,
    #                       3.1754221078376057,
    #                       2.077109999336178,
    #                       2.7644115411788692,
    #                       1087.0125115153648,
    #                       1059.237538387056,
    #                       1099.622198615195,
    #                       1110.2871963523585,
    #                       1143.6329427935093,
    #                       1136.2581148387092,
    #                       1216.1541661034505,
    #                       1121.5768866409626,
    #                       469.8380424441332,
    #                       375.74263912577607,
    #                       8.487788505379974])
    #
    # band_array = (band_array - band_means[:, None, None]) / (band_stds[:, None, None] + 0.001)
    #
    band_tensor = torch.tensor(band_array).float()

    # normalization happens here
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (band_tensor.std(dim=(1, 2)) + 0.01)
    band_tensor = band_tensor.permute(2, 0, 1)

    return band_tensor

class SentinelTiffDataloader(Dataset):
    def __init__(self, training_feature_path, training_labels_path, id_month_list, bands_to_keep,
                 corrupted_transform_method):
        self.training_feature_path = training_feature_path
        self.training_labels_path = training_labels_path
        self.id_month_list = id_month_list
        self.bands_to_keep = bands_to_keep
        self.corrupted_transform_method = corrupted_transform_method

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):
        current_id, month = self.id_month_list[idx]

        label_path = osp.join(self.training_labels_path, f"{current_id}_agbm.tif")
        label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        feature_tensor = retrieve_tiff(self.training_feature_path, current_id, month)

        sample = {'image': feature_tensor, 'label': label_tensor}

        selected_tensor = apply_transforms(bands_to_keep=self.bands_to_keep,
                                           corrupted_transform_method=self.corrupted_transform_method)(sample)

        # return feature_tensor, label_tensor
        # return selected_tensor['image'], selected_tensor['label']
        return selected_tensor['image'], label_tensor


class SentinelTiffDataloader_all(Dataset):
    def __init__(self, training_feature_path, training_labels_path, chip_ids, bands_to_keep, corrupted_transform_method,
                 channel_num, ):
        self.training_feature_path = training_feature_path
        self.training_labels_path = training_labels_path
        self.chip_ids = chip_ids
        self.bands_to_keep = bands_to_keep
        self.corrupted_transform_method = corrupted_transform_method
        self.channel_num = channel_num

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        times = 0
        current_id = self.chip_ids[idx]
        label_path = osp.join(self.training_labels_path, f"{current_id}_agbm.tif")
        # label = rasterio.open(label_path).read().astype(np.float32)
        label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))
        for month in range(5, 12):

            if month < 10:
                month = "0" + str(month)

            month_patch_path = osp.join(self.training_feature_path, f"{current_id}_S2_{month}.tif")
            if osp.exists(month_patch_path):
                # bands_list = retrieve_tiff(self.training_feature_path, current_id, month, self.augment)
                feature_tensor = retrieve_tiff(self.training_feature_path, current_id, month)
                sample = {'image': feature_tensor, 'label': label_tensor}

                selected_tensor = apply_transforms(bands_to_keep=self.bands_to_keep,
                                                   corrupted_transform_method=self.corrupted_transform_method)(sample)
                if times == 0:
                    data_tensor = selected_tensor['image']
                else:
                    data_tensor = torch.cat((data_tensor, selected_tensor['image']), dim=0)
                times += 1

        if data_tensor.shape[0] < self.channel_num:
            indexes = list(range(data_tensor.shape[0]))
            for i in range(self.channel_num - data_tensor.shape[0]):
                random.shuffle(indexes)
                tensor_temp = copy.deepcopy(data_tensor[indexes[0]])
                tensor_temp = tensor_temp.unsqueeze(0)
                data_tensor = torch.cat((data_tensor, tensor_temp), dim=0)

        return data_tensor, label_tensor


class SentinelTiffDataloaderSubmission(Dataset):
    def __init__(self, testing_features_path, id_month_list, bands_to_keep, corrupted_transform_method):
        self.testing_features_path = testing_features_path
        self.id_month_list = id_month_list
        self.bands_to_keep = bands_to_keep
        self.corrupted_transform_method = corrupted_transform_method

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):
        id, month = self.id_month_list[idx]

        feature_tensor = retrieve_tiff(self.testing_features_path, id, month)

        sample = {'image': feature_tensor}

        selected_tensor = apply_transforms_testing(bands_to_keep=self.bands_to_keep,
                                                   corrupted_transform_method=self.corrupted_transform_method)(sample)

        return selected_tensor['image']


class SentinelTiffDataloaderSubmission_all(Dataset):
    def __init__(self, testing_features_path, chip_ids, bands_to_keep, corrupted_transform_method, channel_num):
        self.testing_features_path = testing_features_path
        self.chip_ids = chip_ids
        self.bands_to_keep = bands_to_keep
        self.corrupted_transform_method = corrupted_transform_method
        self.channel_num = channel_num

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        current_id = self.chip_ids[idx]
        times = 0
        for month in range(5, 12):
            if month < 10:
                month = "0" + str(month)
            month_patch_path = osp.join(self.testing_features_path, f"{current_id}_S2_{month}.tif")

            if osp.exists(month_patch_path):
                feature_tensor = retrieve_tiff(self.testing_features_path, current_id, month)

                sample = {'image': feature_tensor}

                selected_tensor = apply_transforms_testing(bands_to_keep=self.bands_to_keep,
                                                           corrupted_transform_method=self.corrupted_transform_method)(
                    sample)
                if times == 0:
                    data_tensor = selected_tensor['image']
                else:
                    data_tensor = torch.cat((data_tensor, selected_tensor['image']), dim=0)
                times += 1

        if data_tensor.shape[0] < self.channel_num:
            indexes = list(range(data_tensor.shape[0]))
            for i in range(self.channel_num - data_tensor.shape[0]):
                random.shuffle(indexes)
                tensor_temp = copy.deepcopy(data_tensor[indexes[0]])
                tensor_temp = tensor_temp.unsqueeze(0)
                data_tensor = torch.cat((data_tensor, tensor_temp), dim=0)

        return data_tensor


def retrieve_tiff(feature_path, id, month) -> torch.Tensor:
    S1_path = osp.join(feature_path, f"{id}_S1_{month}.tif")
    S2_path = osp.join(feature_path, f"{id}_S2_{month}.tif")

    bands = []

    S1_bands = rasterio.open(S1_path).read().astype(np.float32)
    bands.extend(S1_bands)

    S2_bands = rasterio.open(S2_path).read().astype(np.float32)
    bands.extend(S2_bands)

    feature_tensor = create_tensor(bands)

    return feature_tensor


def apply_transforms(corrupted_transform_method, bands_to_keep):
    return nn.Sequential(
        tf.ClampAGBM(vmin=0., vmax=420.),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
        # indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
        # indices.AppendNormalizedDifferenceIndex(index_a=11, index_b=12),  # (VV-VH)/(VV+VH), index 16
        # indices.AppendNDBI(index_swir=8, index_nir=6),
        # # Difference Built-up Index for development detection, index 17
        # indices.AppendNDRE(index_nir=6, index_vre1=3),  # Red Edge Vegetation Index for canopy detection, index 18
        # indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
        # indices.AppendNDWI(index_green=1, index_nir=6),  # Difference Water Index for water detection, index 20
        # indices.AppendSWI(index_vre1=3, index_swir2=8),
        # # Standardized Water-Level Index for water detection, index 21
        # tf.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
        # tf.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
        tf.DropBands(torch.device('cpu'), bands_to_keep),  # DROPS ALL BUT SPECIFIED bands_to_keep
        # corrupted_transform_method  # Applies corrupted band transformation
    )


def apply_transforms_testing(corrupted_transform_method, bands_to_keep):
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
        corrupted_transform_method  # Applies corrupted band transformation
    )
