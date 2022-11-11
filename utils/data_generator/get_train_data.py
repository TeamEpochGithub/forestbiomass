import os
import os.path as osp

from osgeo import gdal  # https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/

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
