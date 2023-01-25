# 12 x (4 + 11) bands in
# missing month = zeros
#
# input = 180 bands
# output = 180 values between 0 - 1
# 	- indicating %corrupted, fully corrupted = 1
# 	- missing = 1
import csv
import os

import torch
from torch.nn import functional as F
import os.path as osp

from tqdm import tqdm

import data
from multiprocessing import Pool

from data import imgs
from models.ensembler.utils import get_bands_else_zeros, create_and_normalize_bands_tensor

path_patch_names = osp.join(osp.dirname(data.__file__), "test_patch_names")  # patch_names

with open(path_patch_names, newline='') as f:
    reader = csv.reader(f)
    patch_name_data = list(reader)
patch_names = patch_name_data[0]

def create_mask(image_tensor, radius):
    assert image_tensor.shape == (256, 256), "Input image must be of shape (256, 256)"
    # add singleton dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Create convolution mask
    convolution_mask = F.avg_pool2d(image_tensor, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    # remove singleton dimension
    convolution_mask = convolution_mask.squeeze(0)
    # Create comparison mask
    comparison_mask = torch.where(image_tensor == convolution_mask, torch.tensor(0.), torch.tensor(1.))
    return comparison_mask[0]


def extract_metadata_band(band, missing):
    if missing:
        return 0
    else:
        return torch.mean(create_mask(band, radius=2))


def get_features_batch(patch_names):
    patch_bands = []
    for patch_name in patch_names:
        path_train_features = osp.join(osp.dirname(imgs.__file__), "test_features")

        bands = []
        for month in range(12):
            S1_path = osp.join(path_train_features, f"{patch_name}_S1_{month:02}.tif")
            S2_path = osp.join(path_train_features, f"{patch_name}_S2_{month:02}.tif")

            bands.extend(get_bands_else_zeros(S1_path, zeros_dim=4))
            bands.extend(get_bands_else_zeros(S2_path, zeros_dim=11))

        feature_tensor = create_and_normalize_bands_tensor(bands)

        patch_bands.append(feature_tensor)
    return patch_bands


def extract_metadata_batch(batch):
    """
    @param batch: A torch tensor of shape (batch_size) containing patch names
    @return: A torch tensor of shape (batch_size, 12 * num_bands) indicating "corruptedness" for each band, where 0 is very corrupted
    and 1 is not corrupted
    """
    patch_bands = get_features_batch(batch)

    batch_metadata = []
    for patch in patch_bands:

        patch_metadata = []
        for band in patch:
            patch_metadata.append(extract_metadata_band(band, False))

        batch_metadata.append(patch_metadata)
    return torch.FloatTensor(batch_metadata)


def save_batch_to_csv(batch_patch_names):
    print(patch_names.index(batch_patch_names[0]))
    corruptedness_values = extract_metadata_batch(batch_patch_names).tolist()
    patch_corrupted = zip(batch_patch_names, corruptedness_values)

    csv_path = (osp.join(osp.dirname(data.__file__), "train_corruptedness_values.csv"))
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(patch_corrupted)


def save_test_batch_to_csv(batch_patch_names):
    print(patch_names.index(batch_patch_names[0]))
    corruptedness_values = extract_metadata_batch(batch_patch_names).tolist()
    patch_corrupted = zip(batch_patch_names, corruptedness_values)

    csv_path = (osp.join(osp.dirname(data.__file__), "test_corruptedness_values.csv"))
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(patch_corrupted)


if __name__ == '__main__':
    batch_size = 8
    patch_names_div = [patch_names[x:x + batch_size] for x in range(0, len(patch_names), batch_size)]

    with Pool(os.cpu_count() - 1) as p:
        p.map(save_test_batch_to_csv, patch_names_div)  # save_batch_to_csv
