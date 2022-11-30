import numpy as np
import os.path as osp


def get_data_for_test(patch_names, train_data_path):
    X_all = []
    y_all = []
    patch_names = patch_names[0:20]
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
    return np.array(X_all), np.array(y_all)
