import csv

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import os.path as osp

import data
from data import imgs

from models.ensembler.utils import get_bands_else_zeros, create_and_normalize_bands_tensor


class UniversalDatasetTrain(Dataset):

    def __init__(self):
        """
        """
        self.path_patch_names = osp.join(osp.dirname(data.__file__), "patch_names")
        self.path_train_features = osp.join(osp.dirname(imgs.__file__), "train_features")
        self.path_train_agbm = osp.join(osp.dirname(imgs.__file__), "train_agbm")

        with open(self.path_patch_names, newline='') as f:
            reader = csv.reader(f)
            patch_name_data = list(reader)
        self.patch_names = patch_name_data[0][:50]

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch = self.patch_names[idx]

        label_path = osp.join(self.path_train_agbm, f"{patch}_agbm.tif")
        label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        bands = []
        for month in range(12):
            S1_path = osp.join(self.path_train_features, f"{patch}_S1_{month:02}.tif")
            S2_path = osp.join(self.path_train_features, f"{patch}_S2_{month:02}.tif")

            bands.extend(get_bands_else_zeros(S1_path, zeros_dim=4))
            bands.extend(get_bands_else_zeros(S2_path, zeros_dim=11))

        feature_tensor = create_and_normalize_bands_tensor(bands)

        return feature_tensor, label_tensor  # (180, 256, 256) , (256, 256)


class UniversalDatasetTest(Dataset):

    def __init__(self):
        """
        """
        self.path_test_patch_names = osp.join(osp.dirname(data.__file__), "test_patch_names")
        self.path_test_features = osp.join(osp.dirname(imgs.__file__), "test_features")

        with open(self.path_test_patch_names, newline='') as f:
            reader = csv.reader(f)
            patch_name_data = list(reader)
        self.test_patch_names = patch_name_data[0]

    def __len__(self):
        return len(self.test_patch_names)

    def __getitem__(self, idx):
        patch = self.test_patch_names[idx]

        bands = []
        for month in range(12):
            S1_path = osp.join(self.path_test_features, f"{patch}_S1_{month:02}.tif")
            S2_path = osp.join(self.path_test_features, f"{patch}_S2_{month:02}.tif")

            bands.extend(get_bands_else_zeros(S1_path, zeros_dim=4))
            bands.extend(get_bands_else_zeros(S2_path, zeros_dim=11))

        feature_tensor = create_and_normalize_bands_tensor(bands)

        return feature_tensor  # (180, 256, 256)


if __name__ == '__main__':
    dataset = UniversalDatasetTrain()

    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample[0].shape, sample[1].shape)
