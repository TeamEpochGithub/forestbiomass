import rasterio
from torch import nn
from torchgeo.transforms import indices

import models.utils.transforms as tf
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import torch
import warnings
import torch.nn.functional as F
from models.utils.transforms import ReplaceClouds

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SentinelDataLoader(Dataset):
    def __init__(self, args, id_month_list, corrupted_transform_method):
        self.training_data_path = args.converted_training_features_path
        self.bands_to_keep = args.bands_to_keep
        self.id_month_list = id_month_list
        self.corrupted_transform_method = corrupted_transform_method
        self.image_patch_size = args.image_patch_size
        assert (
            256 // self.image_patch_size == 256 / self.image_patch_size
        ), "Number has to be integer fraction of 256"
        assert self.image_patch_size % 32 == 0, "Number has to be divisble by 32"
        self.num_patches = (256 // self.image_patch_size) ** 2  # symmetrical

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):
        # this is hell a inefficent way to do this, better to have a list of all the patches and then just index into it but i don't wanna mess with your structure here so keeping it for now
        id, month = self.id_month_list[idx // self.num_patches]

        label_path = osp.join(self.training_data_path, id, "label.npy")
        label = np.load(label_path, allow_pickle=True)
        label_tensor = torch.tensor(np.asarray([label], dtype=np.float32))
        label_tensor = label_tensor

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

        all_tensor = divide_into_patches(all_tensor, self.image_patch_size)
        label_tensor = divide_into_patches(label_tensor, self.image_patch_size)

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
        self.image_patch_size = args.image_patch_size
        assert (
            256 // self.image_patch_size == 256 / self.image_patch_size
        ), "Number has to be integer fraction of 256"
        assert self.image_patch_size % 32 == 0, "Number has to be divisble by 32"
        self.num_patches = (256 // self.image_patch_size) ** 2  # symmetrical

    def __len__(self):
        return len(self.id_month_list)

    def __getitem__(self, idx):
        id, month = self.id_month_list[idx]

        label_tensor = torch.rand((1, self.image_patch_size, self.image_patch_size))

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
            # tf.DropBands(
            #     torch.device("cpu"), self.bands_to_keep
            # ),  # DROPS ALL BUT SPECIFIED bands_to_keep
            self.corrupted_transform_method,  # Applies corrupted band transformation
        )

        all_tensor = divide_into_patches(all_tensor, self.image_patch_size)
        label_tensor = divide_into_patches(label_tensor, self.image_patch_size)

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
    # print(band_tensor.shape)

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
        ReplaceClouds(cloud_channel=10),
    )


def reasamble_patches(patches, kernel_size, stride, batch_size, norm_map):
    patch_shape = patches.shape
    channels = patch_shape[1]
    output_h = 256
    output_w = 256
    a_unf = patches.view(batch_size, -1, channels * kernel_size * kernel_size).permute(
        0, 2, 1
    )
    # print(a_unf.shape)
    patches_orig = F.fold(a_unf, (output_h, output_w), kernel_size, stride=stride)
    patches_orig /= norm_map
    # import matplotlib.pyplot as plt

    # plt.imshow(patches_orig[0, 0, :, :].to(torch.float64))
    # plt.colorbar()
    # plt.show()
    return patches_orig


def divide_into_patches(image, kernel_size):
    if len(image.shape) < 4:
        image.unsqueeze_(0)
    bs = image.shape[0]
    channels = image.shape[1]
    stride = 8
    a_unf = F.unfold(image, kernel_size, padding=0, stride=stride)
    patches = (
        a_unf.permute(0, 2, 1)
        .contiguous()
        .view(bs, -1, channels, kernel_size, kernel_size)
    )
    # print(a_unf.shape)
    # print(patches.shape)
    # norm = F.unfold(torch.ones(image.shape), kernel_size, stride=stride)
    # a_Folded = F.fold(a_unf, image.shape[-2:], kernel_size, stride=stride)
    # norm_map = F.fold(norm, image.shape[-2:], kernel_size, stride=stride)
    # a_Folded /= norm_map
    # import matplotlib.pyplot as plt

    # plt.imshow(image[0, 0, :, :])
    # plt.show()
    # plt.imshow(a_Folded[0, 0, :, :])
    # plt.show()
    # print(torch.allclose(image, a_Folded))
    return patches.squeeze(0)
