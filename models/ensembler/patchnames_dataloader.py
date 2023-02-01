import csv

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import os.path as osp

import data
from data import imgs

from models.ensembler.utils import get_bands_else_zeros, create_and_normalize_bands_tensor


class PatchNameDatasetTrain(Dataset):

    def __init__(self, agbms_dict):
        self.path_patch_names = osp.join(osp.dirname(data.__file__), "patch_names")
        self.path_train_agbm = osp.join(osp.dirname(imgs.__file__), "train_agbm")

        with open(self.path_patch_names, newline='') as f:
            reader = csv.reader(f)
            patch_name_data = list(reader)
        self.patch_names = patch_name_data[0]

        self.agbms_dict = agbms_dict

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch = self.patch_names[idx]

        # label_path = osp.join(self.path_train_agbm, f"{patch}_agbm.tif")
        # label_tensor = torch.tensor(rasterio.open(label_path).read().astype(np.float32))
        label_tensor = torch.tensor(self.agbms_dict[patch])

        return patch  ## label_tensor  # 1, (256, 256)


class PatchNameDatasetTest(Dataset):

    def __init__(self):
        self.path_test_patch_names = osp.join(osp.dirname(data.__file__), "test_patch_names")

        with open(self.path_test_patch_names, newline='') as f:
            reader = csv.reader(f)
            patch_name_data = list(reader)
        self.test_patch_names = patch_name_data[0]

    def __len__(self):
        return len(self.test_patch_names)

    def __getitem__(self, idx):
        patch = self.test_patch_names[idx]

        return patch  # (180, 256, 256)

