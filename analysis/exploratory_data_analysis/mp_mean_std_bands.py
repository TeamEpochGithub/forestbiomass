import csv
from multiprocessing import Pool

import os.path as osp
import data
import csv
import rasterio

from tqdm import tqdm
from models.ensembler.utils import get_bands_else_zeros
import numpy as np


def get_mean_and_std(patch):
    data_path = osp.dirname(data.__file__)
    training_features_path = osp.join(data_path, "imgs", "train_features")

    for month in range(5, 12):
        band_means = []
        band_stds = []

        S1_path = osp.join(training_features_path, f"{patch}_S1_{month:02}.tif")
        S2_path = osp.join(training_features_path, f"{patch}_S2_{month:02}.tif")

        S1_bands = get_bands_else_zeros(S1_path, 4)
        S2_bands = get_bands_else_zeros(S2_path, 11)

        all_bands = np.concatenate((S1_bands, S2_bands), axis=0)

        for band in all_bands:
            band_means.append(np.average(band.flatten()))
            band_stds.append(np.std(band.flatten()))

        with open('band_means.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(band_means)

        with open('band_stds.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(band_stds)


if __name__ == '__main__':
    data_path = osp.dirname(data.__file__)
    training_ids_path = osp.join(data_path, "patch_names")

    with open(training_ids_path, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    # Create a pool of workers
    with Pool() as pool:
        # Apply the function to the input data using map()
        pool.map(get_mean_and_std, tqdm(patch_names))

    # # Print the results
    # print(results)
    # print(np.array(results))
    # print(np.mean(np.array(results), axis=0))
    #
    # # Write the results to a .csv file
    # with open('band_means.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(results)
