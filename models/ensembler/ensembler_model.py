# ensemble_model =
# 	- 180 datapoints in corrupted_model
# 	- 3 weights out of corrupted_model (x1, x2, x3)
#
# prediction =
# 	- x1 * (inference swin_transformer) + x2 * (inference segmenter) + x3 *  (inference pixel_wise)
# 	- then calculate and backpropagate corrupted_model over ensemble_model
import csv
import os
import time

import numpy as np
import rasterio
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
from data import imgs
from models import tb_logs
from models.ensembler.patchnames_dataloader import PatchNameDatasetTrain, PatchNameDatasetTest
from models.utils.discretize_prediction_pixels import discretize_prediction_pixels
from models.utils.loss_functions import rmse_loss
import os.path as osp
import json
from multiprocessing import Pool
from functools import partial

import warnings

# ignore the NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=UserWarning, module='rasterio')

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def bar(curr_step, total_steps, bar_length=30):
    """
    Printing progress bar.
    """
    filled = int(bar_length * curr_step / total_steps)
    return f"[{'#' * filled}{'.' * (bar_length - filled)}]"


def train_and_eval(corrupted_model, train_loader, valid_loader, optimizer, criterion, epochs=10, log_freq=100):
    """
    Train and evaluate the model.

    @param corrupted_model: (torch.nn.Module) model to train
    @param train_loader: (torch.utils.data.DataLoader) train dataloader
    @param valid_loader: (torch.utils.data.DataLoader) validation dataloader
    @param optimizer: (torch.optim) optimizer
    @param criterion: (torch.nn) loss function
    @param epochs: (int) number of epochs
    @return: (list, list) train loss and validation loss
    """
    global_train_loss, global_valid_loss = [], []
    best_train_loss, best_valid_loss = 100, 100

    path_patch_names = osp.join(osp.dirname(data.__file__), "patch_names")
    with open(path_patch_names, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    predictions_dict = create_predictions_dict(["train_swinres_agbm", "train_swinefficientnet_agbm"], patch_names)
    corrupted_metadata_dict = create_metadata_dict("train_corruptedness_values.json")

    for epoch in range(epochs):

        epochtic = time.perf_counter()

        # Training loop
        corrupted_model.train()
        train_loss = []
        for epoch_step, (patch_names_batch, agbms_batch) in enumerate(train_loader):
            # ======== TRAINING ========
            corrupted_metadata = retrieve_metadata_batch(patch_names_batch, corrupted_metadata_dict)

            # 1. Move the tensors to the configured device.
            agbms_batch, corrupted_metadata, corrupted_model = agbms_batch.to(DEVICE), corrupted_metadata.to(
                DEVICE), corrupted_model.to(DEVICE)

            # 2. Forward pass by passing the images through the corrupted_model.
            w1, w2, w3 = torch.tensor_split(corrupted_model(corrupted_metadata), 3, dim=1)
            w1, w2, w3 = w1.view(-1, 1, 1, 1), w2.view(-1, 1, 1, 1), w3.view(-1, 1, 1, 1)
            preds_batch = (w1 * retrieve_predictions_batch("train_swinres_agbm", patch_names_batch,
                                                           predictions_dict).to(
                DEVICE)) + (
                                  w2 * retrieve_predictions_batch("train_swinres_agbm", patch_names_batch,
                                                                  predictions_dict).to(DEVICE)) + (
                                  w3 * retrieve_predictions_batch("train_swinefficientnet_agbm", patch_names_batch,
                                                                  predictions_dict).to(DEVICE))

            # 3. Zero the gradients of all corrupted_model parameters.
            optimizer.zero_grad()
            # 4. Compute the loss.
            loss = criterion(preds_batch, agbms_batch)
            # 5. Backward pass to compute the gradients of the loss w.r.t. the corrupted_model parameters.
            loss.backward()
            # 6. Step the optimizer to update the corrupted_model parameters.
            optimizer.step()
            # =========================

            # Logging metrics
            train_loss.append(loss.item())

            if epoch_step % log_freq == 0 or epoch_step == len(train_loader) - 1:
                print(
                    f"Epoch {epoch + 1:2d}/{epochs:2d} {bar(epoch_step + 1, len(train_loader))} Train-Loss: {np.mean(train_loss):.4f} \r")

        # Evaluation loop
        corrupted_model.eval()
        valid_loss = []
        with torch.no_grad():
            for epoch_step, (patch_names_batch, agbms_batch) in enumerate(valid_loader):

                # ======== VALIDATION ========
                corrupted_metadata = retrieve_metadata_batch(patch_names_batch, corrupted_metadata_dict)

                # 1. Move the tensors to the configured device.
                agbms_batch, corrupted_metadata, corrupted_model = agbms_batch.to(DEVICE), corrupted_metadata.to(
                    DEVICE), corrupted_model.to(DEVICE)

                # 2. Forward pass by passing the images through the corrupted_model.
                w1, w2, w3 = torch.tensor_split(corrupted_model(corrupted_metadata), 3, dim=1)
                w1, w2, w3 = w1.view(-1, 1, 1, 1), w2.view(-1, 1, 1, 1), w3.view(-1, 1, 1, 1)
                preds_batch = (w1 * retrieve_predictions_batch("train_swinres_agbm", patch_names_batch,
                                                               predictions_dict).to(
                    DEVICE)) + (
                                      w2 * retrieve_predictions_batch("train_swinres_agbm", patch_names_batch,
                                                                      predictions_dict).to(DEVICE)) + (
                                      w3 * retrieve_predictions_batch("train_swinefficientnet_agbm", patch_names_batch,
                                                                      predictions_dict).to(DEVICE))

                # 3. Compute the loss.
                loss = criterion(preds_batch, agbms_batch)
                # Note we do not backpropagate the gradients in the validation loop.
                # ==========================

                # Logging metrics
                valid_loss.append(loss.item())

                if epoch_step % log_freq == 0 or epoch_step == len(valid_loader) - 1:
                    print(
                        f"Epoch {epoch + 1:2d}/{epochs:2d} {bar(epoch_step + 1, len(valid_loader))} Valid-Loss: {np.mean(valid_loss):.4f} \r")

        global_train_loss.extend(train_loss)
        global_valid_loss.extend(valid_loss)
        best_train_loss = min(best_train_loss, np.mean(train_loss))
        best_valid_loss = min(best_valid_loss, np.mean(valid_loss))
        print(f"Best Train Loss: {best_train_loss} Best Valid Loss: {best_valid_loss}")

        if epoch % 5 == 0:
            model_save_path = osp.join(osp.dirname(tb_logs.__file__), "ensembler_model",
                                       f"epoch_{epoch}_{np.round(np.mean(train_loss), 1)}_{np.round(np.mean(valid_loss), 1)}_model.pth")
            if DEVICE != "cpu":
                corrupted_model.cpu()
                torch.save(corrupted_model.state_dict(), model_save_path)
                corrupted_model.cuda()
            else:
                torch.save(corrupted_model.state_dict(), model_save_path)

        epochtoc = time.perf_counter()

        print(f"Epoch time: {epochtoc - epochtic:0.4f} seconds")

    return global_train_loss, global_valid_loss


def create_metadata_model(metadata_dim, n_hidden, weights_dim):
    return nn.Sequential(nn.Linear(metadata_dim, 12),  # (metadata_dim, 90)
                         nn.ReLU(),
                         # nn.Linear(90, 12),
                         # nn.ReLU(),
                         nn.Linear(12, weights_dim),
                         nn.Softmax(1))


def create_train_val_dataloaders(num_workers, batch_size, agbms_dict):
    print("Creating dataloaders...")

    dataset = PatchNameDatasetTrain(agbms_dict)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, valid_size])

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("Done!")
    return dataloader_train, dataloader_val


def retrieve_predictions_batch(model_name, patch_names, predictions_dict):
    preds = []
    for patch_name in patch_names:
        # prediction_path = osp.join(osp.dirname(imgs.__file__), model_name, f"{patch_name}_agbm.tif")
        # preds.append(rasterio.open(prediction_path).read().astype(np.float32))
        preds.append(predictions_dict[model_name][patch_name])
    preds = torch.from_numpy(np.array(preds))
    return preds


def create_predictions_dict(model_names, patch_names):
    print("Creating predictions dictionary...")

    with Pool() as pool:
        predictions_dict = {
            model_name: dict(pool.map(load_predictions, [(model_name, patch_name) for patch_name in patch_names])) for
            model_name in model_names}
    print("Done!")

    return predictions_dict


def create_metadata_dict(filename):
    print("Creating metadata dictionary...")
    with open(osp.join(osp.dirname(data.__file__), filename), "r") as f:
        content = f.read()
        metadata_dict = json.loads(content)
    print("Done!")
    return metadata_dict


def load_predictions(args):
    model_name, patch_name = args
    prediction_path = osp.join(osp.dirname(imgs.__file__), model_name, f"{patch_name}_agbm.tif")
    return patch_name, rasterio.open(prediction_path).read().astype(np.float32)


def retrieve_metadata_batch(patch_names, metadata_dict):
    metadatas = []
    for patch_name in patch_names:
        metadatas.append(metadata_dict[patch_name])
    metadatas = np.array(metadatas)
    return torch.Tensor(metadatas)  # (batch_size, 180)


def create_agbms_dict():
    print("Creating agbm dictionary...")

    path_patch_names = osp.join(osp.dirname(data.__file__), "patch_names")
    with open(path_patch_names, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    with Pool() as pool:
        agbms_dict = dict(pool.map(load_predictions, [("train_agbm", patch_name) for patch_name in patch_names]))

    # agbms_dict = {}
    # for patch in patch_names:
    #     label_path = osp.join(osp.join(osp.dirname(imgs.__file__), "train_agbm"), f"{patch}_agbm.tif")
    #     label = rasterio.open(label_path).read().astype(np.float32)
    #     agbms_dict[patch] = label
    print("Done!")

    return agbms_dict


from functools import partial


def create_submission(model_name, discretize_pixels, discrete_pixel_values):
    print("Creating submission...")
    path_patch_names = osp.join(osp.dirname(data.__file__), "test_patch_names")
    with open(path_patch_names, newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    model_names = ["test_swinres_agbm", "test_swinefficientnet_agbm"]

    model_save_path = osp.join(osp.dirname(tb_logs.__file__), "ensembler_model", model_name)
    corruption_model = create_metadata_model(180, 1, 3)
    corruption_model.load_state_dict(torch.load(model_save_path))
    corruption_model.eval()

    test_corrupted_metadata_dict = create_metadata_dict("test_corruptedness_values.json")
    test_predictions_dict = create_predictions_dict(model_names, patch_names)

    test_dataloader = DataLoader(PatchNameDatasetTest(), batch_size=1)

    with Pool(os.cpu_count() - 1) as pool:
        pool.map(partial(discretize_and_save, test_corrupted_metadata_dict=test_corrupted_metadata_dict,
                         model=corruption_model, test_predictions_dict=test_predictions_dict,
                         discretize_pixels=discretize_pixels, discrete_pixel_values=discrete_pixel_values),
                 test_dataloader)

    print("Done!")


def discretize_and_save(patch_names_batch, test_corrupted_metadata_dict, model, test_predictions_dict,
                        discretize_pixels, discrete_pixel_values):
    metadata = retrieve_metadata_batch(patch_names_batch, test_corrupted_metadata_dict)

    # 1. Move the tensors to the configured device.
    metadata, model = metadata.to(DEVICE), model.to(DEVICE)

    # 2. Forward pass by passing the images through the corrupted_model.
    w1, w2, w3 = torch.tensor_split(model(metadata), 3, dim=1)
    w1, w2, w3 = w1.view(-1, 1, 1, 1), w2.view(-1, 1, 1, 1), w3.view(-1, 1, 1, 1)
    preds_batch = (w1 * retrieve_predictions_batch("test_swinres_agbm", patch_names_batch,
                                                   test_predictions_dict).to(DEVICE)) \
                  + (w2 * retrieve_predictions_batch("test_swinres_agbm", patch_names_batch,
                                                     test_predictions_dict).to(DEVICE)) \
                  + (w3 * retrieve_predictions_batch("test_swinefficientnet_agbm", patch_names_batch,
                                                     test_predictions_dict).to(DEVICE))

    test_agbm_path = osp.join(osp.dirname(imgs.__file__), "test_ensemble_agbm", f"{patch_names_batch[0]}_agbm.tif")

    pred_array = np.asarray(preds_batch[0][0].cpu().detach().numpy())
    if discretize_pixels:
        pred_array = map_to_closest_values(pred_array, discrete_pixel_values)

    im = Image.fromarray(pred_array)
    im.save(test_agbm_path)
    print(f"patch {patch_names_batch[0]} saved")


def map_to_closest_values(array, discrete_values):
    idx = np.argmin(np.abs(discrete_values - array.flatten()[:, np.newaxis]), axis=1)
    return np.take(discrete_values, idx).reshape(array.shape)


if __name__ == '__main__':
    # metadata_dim = 180
    # n_hidden = 180
    # weights_dim = 3
    # learning_rate = 1e-4
    # corrupted_metadata_model = create_metadata_model(metadata_dim=metadata_dim, n_hidden=n_hidden, weights_dim=weights_dim)
    # optimizer = torch.optim.SGD(corrupted_metadata_model.parameters(), lr=learning_rate)
    # loss_function = rmse_loss
    # epochs = 100
    # num_workers = 20  # up to 24
    #
    # batch_size = 16
    # image_size = 256
    # placeholder = (lambda x: torch.rand((x.shape[0], image_size, image_size)))
    #
    # agbms_dict = create_agbms_dict()
    # train_loader, valid_loader = create_train_val_dataloaders(num_workers=num_workers, batch_size=batch_size,
    #                                                           agbms_dict=agbms_dict)
    # warm_start = True
    # if warm_start:
    #     model_save_path = osp.join(osp.dirname(tb_logs.__file__), "ensembler_model", "epoch_5_30.6_30.1_model.pth")
    #     corrupted_metadata_model.load_state_dict(torch.load(model_save_path))
    #
    # train_and_eval(corrupted_model=corrupted_metadata_model,
    #                train_loader=train_loader,
    #                valid_loader=valid_loader,
    #                optimizer=optimizer,
    #                criterion=loss_function,
    #                epochs=epochs, log_freq=5)

    discretize_pixels = False
    if discretize_pixels:
        with open(osp.join(osp.dirname(data.__file__), 'label_discrete_pixels'), newline='') as f:
            reader = csv.reader(f)
            discrete_pixel_data = list(reader)
        discrete_pixel_values = sorted(map(lambda x: float(x), list(set(discrete_pixel_data[0]))))
    else:
        discrete_pixel_values = []

    create_submission(model_name="epoch_5_30.6_30.1_model.pth", discretize_pixels=discretize_pixels,
                      discrete_pixel_values=discrete_pixel_values)
