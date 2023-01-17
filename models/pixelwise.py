import csv
import math

import numpy as np
import torch
from keras.layers import Dense
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Sequential, Dropout
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import data
from models.utils import loss_functions
from models.utils.dataloading import create_tensor, apply_transforms
import pytorch_lightning as pl


class PixelWiseModel(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()

        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))

        self.model = model
        self.learning_rate = 1e-4
        self.train_loss_function = loss_functions.rmse_loss
        self.val_loss_function = loss_functions.rmse_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.train_loss_function(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.val_loss_function(y_hat, y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        return self.model(x)


def train_pixel_wise():
    train_dataset = PixelWiseDataLoader()

    validation_fraction = 0.2
    batch_size = 32
    dataloader_workers = 4
    epochs = 20
    log_step_frequency = 10
    model_identifier = "pixel_wise"

    train_size = int(1 - validation_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=dataloader_workers)
    valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                  num_workers=dataloader_workers)

    logger = TensorBoardLogger("pixel_wise_logs", name=model_identifier)

    model = PixelWiseModel(input_dim=4500)
    trainer = Trainer(
        max_epochs=epochs,
        logger=[logger],
        log_every_n_steps=log_step_frequency,
        # callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=[1],
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    return model, str(trainer.callback_metrics['val/loss'].item())


class PixelWiseDataLoader(Dataset):
    def __init__(self):
        with open(osp.join(osp.dirname(data.__file__), "patch_names"), newline='\n') as f:
            reader = csv.reader(f)
            patch_name_data = list(reader)

        self.patch_names = patch_name_data[0]
        self.training_data_path = str(osp.join(osp.dirname(data.__file__), "converted"))

    def __len__(self):
        return len(self.patch_names * 65536)

    def __getitem__(self, idx):

        patch_name = self.patch_names[int(idx // 65536)]
        patch_specific_index = idx % 65536
        row, col = patch_specific_index % 256, math.floor(patch_specific_index / 256)

        all_list = []

        for month in range(12):

            for index in range(11):
                month_patch_path = osp.join(self.training_data_path, patch_name, str(month), "S2", f"{index}.npy")
                if osp.exists(month_patch_path):
                    band = np.load(month_patch_path, allow_pickle=True)
                    all_list.append(band)
                else:
                    all_list.append(np.zeros(256, 256))

            for index in range(4):
                month_patch_path = osp.join(self.training_data_path, patch_name, str(month), "S1", f"{index}.npy")
                if osp.exists(month_patch_path):
                    band = np.load(month_patch_path, allow_pickle=True)
                    all_list.append(band)
                else:
                    all_list.append(np.zeros(256, 256))

        all_tensor = create_tensor(all_list)

        patch_vals = []
        for band in all_tensor:
            patch_vals += get_surrounding_pixels(band, (row, col), 2)

        label_path = osp.join(self.training_data_path, patch_name, "label.npy")
        label = np.load(label_path, allow_pickle=True)[row, col]

        return patch_vals, label


def get_surrounding_pixels(image, index, n):
    """
    Returns the n surrounding (vertically and horizontally) pixels of the specified
    index in the given image. If a pixel is not within the bounds of the image, a zero
    is added to the list of surrounding pixels.

    Parameters:
    - image: a numpy array representing the image
    - index: a tuple of the form (row, column) representing the index of the pixel
    - n: an integer representing the number of surrounding pixels to extract

    Returns:
    - A list of pixel values, each value representing the value of a surrounding pixel,
      or a zero if the pixel is not within the bounds of the image
    """
    # Unpack the index
    row, col = index

    # Initialize an empty list to store the surrounding pixels
    surrounding_pixels = []

    # Calculate the range of rows and columns to iterate over
    row_range = range(row - n, row + n + 1)
    col_range = range(col - n, col + n + 1)

    # Iterate over the rows and columns
    for i in row_range:
        for j in col_range:
            # Check if the pixel is within the bounds of the image
            if 0 <= i < image.shape[0] and 0 <= j < image.shape[1]:
                # If it is, append its value to the list of surrounding pixels
                surrounding_pixels.append(image[i, j])
            else:
                # If it is not, append a zero to the list of surrounding pixels
                surrounding_pixels.append(0)

    return surrounding_pixels
