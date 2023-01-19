import numpy as np
import rasterio
import torch
import os.path as osp


def create_and_normalize_bands_tensor(band_list):
    band_array = np.asarray(band_list, dtype=np.float32)

    band_tensor = torch.tensor(band_array)

    # normalization happens here
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (band_tensor.std(dim=(1, 2)) + 0.01)
    band_tensor = band_tensor.permute(2, 0, 1)

    return band_tensor


def get_bands_else_zeros(path, zeros_dim):
    bands = np.zeros((zeros_dim, 256, 256))  # Missing months replaced with zeros
    if osp.exists(path):
        bands = rasterio.open(path).read().astype(np.float32)
    return bands
