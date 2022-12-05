from collections import Counter

import numpy as np
from models.utils.get_train_data import get_all_bands
import os.path as osp
import csv
import data
import math


def discretize_prediction_pixels(prediction):
    """
    Returns a prediction's pixels by closest counterparts in discrete pixel list
    """
    # discrete_pixel_values = get_discrete_pixel_list()
    with open(osp.join(osp.dirname(data.__file__), 'label_discrete_pixels'), newline='') as f:
        reader = csv.reader(f)
        discrete_pixel_data = list(reader)
    discrete_pixel_values = sorted(map(lambda x: float(x), list(set(discrete_pixel_data[0]))))

    for row_ind, row in enumerate(prediction):
        for col_ind, col in enumerate(prediction):
            prediction[row_ind][col_ind] = find_nearest(list(discrete_pixel_values), prediction[row_ind][col_ind])

    return prediction


def write_discrete_pixels():
    """
    Immediately writes pixels to .csv so that image bands fit in RAM
    """
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]
    patch_names = patch_names[0:5]

    train_data_path = osp.join(osp.dirname(data.__file__), "forest-biomass")
    final_discrete_pixels = []
    for patch in patch_names:
        X_all, y_all, selected_patch_names = get_all_bands([patch], train_data_path)

        with open(osp.join(osp.dirname(data.__file__), 'label_discrete_pixels'), newline='') as f:
            reader = csv.reader(f)
            discrete_pixel_data = list(reader)
        discrete_pixels = discrete_pixel_data[0]
        discrete_pixels = set(map(lambda x: round(float(x), 2), discrete_pixels))

        for label in y_all:
            label_pixels = set(map(lambda x: round(float(x), 2), label.flatten()))
            discrete_pixels.update(label_pixels)
        final_discrete_pixels = list(set(discrete_pixels))

        with open(osp.join(osp.dirname(data.__file__), 'label_discrete_pixels'), 'w', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(list(final_discrete_pixels))

    return sorted(final_discrete_pixels)


def get_discrete_pixel_list():
    """
    Returns list of pixel values found in agbm labels, which turned out to be discrete,
    WARNING: Does not fit in RAM if used on a lot of patches
    """
    train_data_path = osp.join(osp.dirname(data.__file__), "forest-biomass")
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    X_all, y_all, selected_patch_names = get_all_bands(patch_names, train_data_path)
    discrete_pixel_values = set()
    for label in y_all:
        discrete_pixel_values.update(set(label.flatten()))

    return discrete_pixel_values


def find_nearest(array, value):
    """
    Given an array of values and a value x it returns the value in the array closest to x
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


if __name__ == '__main__':
    res = write_discrete_pixels()
    print(res)
    print(len(res))
