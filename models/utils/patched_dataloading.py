import rasterio
from torch import nn
from torchgeo.transforms import indices

import models.utils.transforms as tf
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import torch
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SentinelDataLoader(Dataset):
    def __init__(self, args, id_month_list, corrupted_transform_method):
        self.training_data_path = args.converted_training_features_path
        self.bands_to_keep = args.bands_to_keep
        self.id_month_list = id_month_list
        self.corrupted_transform_method = corrupted_transform_method
        assert (
            256 // self.image_size == 256 / self.image_size,
            "Number has to be integer fraction of 256",
        )
        assert (self.image_size % 32 == 0, "Number has to be divisble by 32")
        self.image_batch_size = args.image_batch_size
        self.num_batches = 256 // self.image_batch_size

    def __len__(self):
        return len(self.id_month_list) * self.num_batche

    def __getitem__(self, idx):

        id, month = self.id_month_list[idx]

        label_path = osp.join(self.training_data_path, id, "label.npy")
        label = np.load(label_path, allow_pickle=True)
        label_tensor = torch.tensor(np.asarray([label], dtype=np.float32))
        label_tensor = label_tensor[:, :32, :32]

        all_list = []

        for index in range(11):
            band = np.load(
                osp.join(self.training_data_path, id, month, "S2", f"{index}.npy"),
                allow_pickle=True,
            )
            all_list.append(band)

        for index in range(4):
            band = np.load(
                osp.join(self.training_data_path, id, month, "S1", f"{index}.npy"),
                allow_pickle=True,
            )
            all_list.append(band)

        all_tensor = create_tensor(all_list)

        sample = {
            "image": all_tensor,
            "label": label_tensor,
        }  # 'image' and 'label' are used by torchgeo
        selected_tensor = apply_transforms(
            bands_to_keep=self.bands_to_keep,
            corrupted_transform_method=self.corrupted_transform_method,
        )(sample)
        return selected_tensor["image"], selected_tensor["label"]


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

        label_tensor = torch.rand((32, 32))

        all_list = []

        for index in range(11):
            band = np.load(
                osp.join(self.training_data_path, id, month, "S2", f"{index}.npy"),
                allow_pickle=True,
            )
            all_list.append(band)

        for index in range(4):
            band = np.load(
                osp.join(self.training_data_path, id, month, "S1", f"{index}.npy"),
                allow_pickle=True,
            )
            all_list.append(band)

        all_tensor = create_tensor(all_list)

        transforms = nn.Sequential(
            # tf.ClampAGBM(vmin=0., vmax=500.),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
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
            # Standardized Water-Level Index for water detection, index 21
            tf.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
            tf.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
            tf.DropBands(
                torch.device("cpu"), self.bands_to_keep
            ),  # DROPS ALL BUT SPECIFIED bands_to_keep
            self.corrupted_transform_method,  # Applies corrupted band transformation
        )

        sample = {
            "image": all_tensor,
            "label": label_tensor,
        }  # 'image' and 'label' are used by torchgeo
        selected_tensor = transforms(sample)
        return selected_tensor["image"]


def create_tensor(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)

    band_tensor = torch.tensor(band_array)

    # normalization happens here
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (
        band_tensor.std(dim=(1, 2)) + 0.01
    )
    band_tensor = band_tensor.permute(2, 0, 1)
    band_tensor = band_tensor[:, :32, :32]
    print(band_tensor.shape)

    return band_tensor


def apply_transforms(corrupted_transform_method, bands_to_keep):
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
        # Standardized Water-Level Index for water detection, index 21
        tf.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
        tf.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
        tf.DropBands(
            torch.device("cpu"), bands_to_keep
        ),  # DROPS ALL BUT SPECIFIED bands_to_keep
        corrupted_transform_method,  # Applies corrupted band transformation
    )