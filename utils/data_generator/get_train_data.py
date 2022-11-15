import csv
import os
import os.path as osp

from osgeo import gdal  # https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/
import numpy
import numpy as np
import osgeo.gdal

import data
import data.imgs as img_data
from osgeo import gdal, ogr  # https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/
import data.imgs as img_data

gdal.PushErrorHandler('CPLQuietErrorHandler')


def get_all_patches() -> dict:
    """
    Get all patches from training data
    :return: Dictionary with patch name as key, and values: dictionaries with individual month+satellite data as keys
    """
    train_data_path = osp.join(osp.dirname(img_data.__file__), "train_features")
    directory = os.fsencode(train_data_path)

    current_filename = "XXXXXXXX"

    all_data = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if not filename.startswith(current_filename):  # If we reach a different patch, increase the counter
            current_filename = filename[:8]
            all_data[current_filename] = {}
        else:
            img_path = osp.join(train_data_path, filename)
            dataset = gdal.Open(img_path)
            data = dataset.ReadAsArray()
            all_data[current_filename][filename] = data

    return all_data


def get_n_patches(n: int) -> dict:
    """
    Get n patches from training data
    :param n: The amount of patches to retrieve data from
    :return: Dictionary with patch name as key, and values: dictionaries with individual month+satellite data as keys
    """
    train_data_path = osp.join(osp.dirname(img_data.__file__), "train_features")
    directory = os.fsencode(train_data_path)

    current_filename = "XXXXXXXX"
    counter = -1  # We start at -1 because first current_filename should not count

    all_data = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if not filename.startswith(current_filename):  # If we reach a different patch, increase the counter
            counter += 1
            if counter >= n:
                break

            current_filename = filename[:8]
            all_data[current_filename] = {}
        else:
            img_path = osp.join(train_data_path, filename)
            dataset = gdal.Open(img_path)
            data = dataset.ReadAsArray()
            all_data[current_filename][filename] = data

    return all_data


def convert_path_to_gdal_dataset(path: str) -> osgeo.gdal.Dataset:
    dataset = gdal.Open(path)
    return dataset


def convert_gdal_dataset_to_ndarray(dataset: osgeo.gdal.Dataset) -> numpy.ndarray:
    data = dataset.ReadAsArray()
    return data


def save_ndarray(relative_path: str, name: str, numpy_data: numpy.ndarray):
    path = osp.join(osp.dirname(data.__file__), relative_path)
    if not (os.path.exists(path)):
        os.makedirs(path)

    np.save(osp.join(path, name), numpy_data)


def get_data_from_path(path: str):
    train_data_path = osp.join(osp.dirname(img_data.__file__), "train_features")
    img_path = osp.join(train_data_path, path)
    dataset = convert_path_to_gdal_dataset(img_path)
    if dataset is None:
        return None
    data = convert_gdal_dataset_to_ndarray(dataset)
    return data


def extract_and_save_patch_names():
    """
    Get all patch names from training data and then save them
    """
    train_data_path = osp.join(osp.dirname(img_data.__file__), "train_features")

    directory = os.fsencode(train_data_path)

    current_patchname = "XXXXXXXX"
    all_patch_names = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if not filename.startswith(current_patchname):
            current_patchname = filename[:8]
            all_patch_names.append(current_patchname)

    with open(osp.join(osp.dirname(data.__file__, 'patch_names')), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(all_patch_names)


def save_all_patches():
    with open(osp.join(osp.dirname(data.__file__, 'patch_names')), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    train_data_path = osp.join(osp.dirname(img_data.__file__), "train_features")
    train_label_path = osp.join(osp.dirname(img_data.__file__), "train_agbm")

    for patch in patch_names:
        print(patch)

        label_path = osp.join(train_label_path, patch + "_agbm.tif")
        label_data = get_data_from_path(label_path)
        save_label_path = osp.join("converted", patch)

        save_ndarray(save_label_path, "label", label_data)

        for month in range(0, 12):
            print(month)

            s1_path = osp.join(train_data_path, patch + "_S1_" + f"{month:02}.tif")
            s2_path = osp.join(train_data_path, patch + "_S2_" + f"{month:02}.tif")

            data_s1 = get_data_from_path(s1_path)
            data_s2 = get_data_from_path(s2_path)

            if data_s1 is not None:
                for band, d in enumerate(data_s1):
                    save_s1_path = osp.join("converted", patch, str(month), "S1")
                    save_ndarray(save_s1_path, str(band), d)

            if data_s2 is not None:
                for band, d in enumerate(data_s2):
                    save_s2_path = osp.join("converted", patch, str(month), "S2")
                    save_ndarray(save_s2_path, str(band), d)