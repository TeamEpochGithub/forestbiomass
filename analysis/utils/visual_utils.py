import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from models.transformer import create_vit_model, fit_vit_model
from models.utils.get_train_data import get_average_green_band_data
from models.utils.root_mean_squared_error import root_mean_squared_error
import os.path as osp
import csv
import data
from osgeo import gdal  # https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/

def visualize_all_bands_patch(patch_name, train_data_path):
    months = []
    for month in range(0, 12):
        bands = []
        for band in range(0, 11):
            try:
                patch_data = np.load(osp.join(train_data_path, patch_name, str(month), "S2", f"{band}.npy"), allow_pickle=True)
                bands.append(patch_data)
            except IOError as e:
                bands.append(np.full((256, 256), -6666))

        months.append(bands)

    columns = 11
    rows = 12
    fig = plt.figure(figsize=(8, 8))
    c = 1
    for month in months:
        for band in month:
            fig.add_subplot(rows, columns, c)
            plt.imshow(band)
            # plt.colorbar()
            plt.axis("off")
            c += 1
    plt.show()

def plot_pred_and_label(pred, label, score, patch_name):
    plt.subplot(2, 2, 1)
    plt.imshow(pred)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(label)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(pred, vmin=0, vmax=500)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(label, vmin=0, vmax=500)
    plt.colorbar()

    plt.suptitle("RMSE:" + str(score) + " of patch: " + patch_name)
    plt.show()


def plot_patch_name(patch_name, train_data_path):
    for month in range(0, 12):
        try:
            patch_data = np.load(osp.join(train_data_path, patch_name, str(month), "S2", "1.npy"), allow_pickle=True)
        except IOError as e:
            continue

        plt.imshow(patch_data)
        plt.colorbar()
        plt.show()


def plot_patch_data(patch_data):
    plt.imshow(patch_data)
    plt.colorbar()
    plt.show()


def plot_patch_path(patch_path, train_data_path):
    for month in range(0, 12):
        try:
            patch_data = np.load(osp.join(train_data_path, patch_path), allow_pickle=True)
        except IOError as e:
            continue

        plt.imshow(patch_data)
        plt.colorbar()
        plt.show()


def flatten_image_from_path(path, train_data_path):
    img_path = osp.join(train_data_path, path)
    dataset = gdal.Open(img_path)
    data = dataset.ReadAsArray()
    return np.average(data, axis=0)


def flatten_agbm_from_path(path, train_label_path):
    img_path = osp.join(train_label_path, path)
    dataset = gdal.Open(img_path)
    data = dataset.ReadAsArray()
    return data


def calculate_mse(path1, path2):
    month0 = flatten_image_from_path(path1)
    month1 = flatten_image_from_path(path2)

    mse = (np.square(month0 - month1)).mean(axis=0)
    return mse


def calculate_average_pixels(paths):
    imgs = []
    for p in paths:
        imgs.append(flatten_image_from_path(p))
    return np.average(imgs, axis=0)


def bar_plot(ax, data, plot_name, x_label, y_label, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), loc='lower right')

    ax.set_title(plot_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
