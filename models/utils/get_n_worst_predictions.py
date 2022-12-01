import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

from models.transformer import create_vit_model, fit_vit_model
from models.utils.get_test_data import get_data_for_test
from models.utils.root_mean_squared_error import root_mean_squared_error
import os.path as osp
import csv
import data


def get_n_worst(n, predictions, labels) -> list:
    """
    Calculate the worst predictions by comparing them to labels based on RMSE, returns prediction, label, and score
    """
    pred_label_score = []
    for ind, pred in enumerate(predictions):
        label = labels[ind]
        pred_label_score.append((pred, label, root_mean_squared_error(y_true=pred, y_pred=label)))
    worst = sorted(pred_label_score, key=lambda x: x[2], reverse=True)
    return worst[0:n]


def get_worst_predictions_from_model(model, n=10) -> list:
    """
    Calculate the worst predictions from a model based on RMSE, returns prediction, label, and score
    """
    train_data_path = osp.join(osp.dirname(data.__file__), "forest-biomass")
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    X_all, y_all, selected_patch_names = get_data_for_test(patch_names, train_data_path)
    X_all = X_all.reshape(X_all.shape[0], 256, 256, 1)
    # x_test = x_test.reshape(x_test.shape[0], 256, 256, 1)

    pred_label_score = []
    for ind, patch in enumerate(X_all):
        label = y_all[ind]
        pred = model.predict(np.array([patch.clip(min=0, max=300), ]))[0]
        pred_label_score.append((pred, label, root_mean_squared_error(y_true=pred, y_pred=label.clip(min=0, max=300)), selected_patch_names[ind]))

    worst = sorted(pred_label_score, key=lambda x: x[2], reverse=True)

    return worst[0:n]


if __name__ == '__main__':
    print("Getting train data...")
    train_data_path = osp.join(osp.dirname(data.__file__), "forest-biomass")
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    X_all, y_all = get_data_for_test(patch_names, train_data_path)

    x_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    x_train = x_train.reshape(x_train.shape[0], 256, 256, 1)
    x_test = x_test.reshape(x_test.shape[0], 256, 256, 1)
    print("Done!")

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    projection_dim = 16
    patch_size = 16  # 6  # Size of the patches to be extract from the input images
    image_size = 256  # 72  # We'll resize input images to this size
    num_patches = (image_size // patch_size) ** 2
    model = create_vit_model(input_shape=(256, 256, 1),
                             x_train=x_train,
                             patch_size=patch_size,
                             num_patches=num_patches,
                             projection_dim=projection_dim,
                             transformer_layers=16,
                             num_heads=4,
                             transformer_units=[projection_dim * 2, projection_dim, ],
                             mlp_head_units=[4096, 2048, 1024],
                             learning_rate=0.001,
                             weight_decay=0.0001)

    fitted_model = fit_vit_model(model=model, x_train=x_train, y_train=y_train, batch_size=16,
                                 num_epochs=2, validation_split=0.1, save_checkpoint=True)

    worst_n = get_worst_predictions_from_model(fitted_model, 5)
    print(worst_n[0])
