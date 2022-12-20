import numpy as np
import os.path as osp


def get_all_from_band(patch_names, band, train_data_path):
    X_all = []

    for patch in patch_names:
        for month in range(0, 12):
            month_patch_path = osp.join(train_data_path, patch, str(month), "S2", f"{band}.npy")
            try:
                month_patch = np.load(month_patch_path, allow_pickle=True)
                X_all.append(month_patch)
            except IOError as e:
                continue

    return np.array(X_all)


def get_average_green_band_data(patch_names, train_data_path):
    X_all = []
    y_all = []
    selected_patch_names = []

    for patch in patch_names:
        label_path = osp.join(train_data_path, patch, "label.npy")

        try:
            label = np.load(label_path, allow_pickle=True)
            if label.shape == ():
                continue
        except IOError as e:
            continue

        month_data = []
        for month in range(0, 12):
            month_patch_path = osp.join(train_data_path, patch, str(month), "S2",
                                        "1.npy")  # 1 is the green band, out of the 11 bands

            try:
                month_patch = np.load(month_patch_path, allow_pickle=True)
                month_data.append(month_patch)
            except IOError as e:
                continue
        if len(month_data) >= 1:
            X_all.append(np.average(month_data, axis=0))
            y_all.append(label)
            selected_patch_names.append(patch)
    return np.array(X_all), np.array(y_all), selected_patch_names


def get_all_bands(patch_names, train_data_path):
    X_all = []
    y_all = []
    selected_patch_names = []

    for patch in patch_names:
        label_path = osp.join(train_data_path, patch, "label.npy")

        try:
            label = np.load(label_path, allow_pickle=True)
            if label.shape == ():
                continue
        except IOError as e:
            continue

        for month in range(0, 12):
            month_bands = []
            month_available = True

            for band in range(0, 11):
                patch_month_band_path = osp.join(train_data_path, patch, str(month), "S2",
                                                 f"{band}.npy")

                try:
                    patch_month_band_data = np.load(patch_month_band_path, allow_pickle=True)
                    month_bands.append(patch_month_band_data)
                except IOError as e:
                    month_available = False

            if not month_available:
                continue

            X_all.append(np.array(month_bands))
            y_all.append(label)
            selected_patch_names.append(patch)

    return np.array(X_all), np.array(y_all), selected_patch_names


def get_data_for_segmenter(patch_names, train_data_path):
    X_all = []
    y_all = []

    for patch in patch_names:
        label_path = osp.join(train_data_path, patch, "label.npy")

        try:
            label = np.load(label_path, allow_pickle=True)
            if label.shape == ():
                continue
        except IOError as e:
            continue

        for month in range(0, 12):
            month_data = []

            for band in range(0, 11):
                month_patch_band_path = osp.join(train_data_path, patch, str(month), "S2",
                                                 f"{str(band)}.npy")

                try:
                    month_patch_band = np.load(month_patch_band_path, allow_pickle=True)
                    month_data.append(month_patch_band)
                except IOError as e:
                    continue

            X_all.append(month_data)
            y_all.append(label)

    return np.array(X_all), np.array(y_all)
