import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import segmentation_models_pytorch as smp
import os
import rasterio
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import warnings
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
import os.path as osp
from pytorch_lightning.loggers import TensorBoardLogger
import data
import csv
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class Segmentation_Dataset_Maker(Dataset):
    def __init__(self, tensor_list, transform=None):
        self.tensor_list = tensor_list
        self.length = len(tensor_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        data_tensor, label_tensor = self.tensor_list[idx]

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label_tensor


class Sentinel2Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.0001)]

    def forward(self, x):
        return self.model(x)


def prepare_dataset(chip_ids, train_data_path):

    # Change value to 1 to include band during training:

    sentinel_1_bands = {
        "VV ascending": 0,
        "VH ascending": 0,
        "VV descending": 0,
        "VH descending": 0
    }

    sentinel_2_bands = {
        "B2-Blue": 1,
        "B3-Green": 1,
        "B4-Red": 1,
        "B5-Veg red edge 1": 1,
        "B6-Veg red edge 2": 1,
        "B7-Veg red edge 3": 1,
        "B8-NIR": 1,
        "B8A-Narrow NIR": 1,
        "B11-SWIR 1": 1,
        "B12-SWIR 2": 1,
        "Cloud probability": 0
    }

    s1_list = list(sentinel_1_bands.values())
    s2_list = list(sentinel_2_bands.values())

    number_of_channels = s1_list.count(1) + s2_list.count(1)

    available_tensors = []

    for id in chip_ids:

        label_path = osp.join(train_data_path, id, "label.npy")

        try:
            label = np.load(label_path, allow_pickle=True)
            if label.shape == ():
                continue
        except IOError as e:
            continue

        label_tensor = torch.tensor(label)

        for month in range(0, 12):

            month_patch_path = osp.join(train_data_path, id, str(month))  # 1 is the green band, out of the 11 bands

            try:

                arr_list = []

                for index, s1_index in enumerate(s1_list):

                    if s1_index == 1:

                        band = np.load(osp.join(month_patch_path, "S1", f"{index}.npy"), allow_pickle=True)
                        arr_list.append(band)

                for index, s2_index in enumerate(s2_list):

                    if s2_index == 1:
                        band = np.load(osp.join(month_patch_path, "S2", f"{index}.npy"), allow_pickle=True)
                        arr_list.append(band)

                data_tensor = torch.tensor(arr_list)

                # These operations were present in the original notebook. I'll figure out some other time what they do.
                # By some other time I mean never.
                data_tensor = (data_tensor.permute(1, 2, 0) - data_tensor.mean(dim=(1, 2))) / (data_tensor.std(dim=(1, 2)) + 0.01)
                data_tensor = data_tensor.permute(2, 0, 1)

                available_tensors.append((data_tensor, label_tensor))

            except IOError as e:
                continue

    new_dataset = Segmentation_Dataset_Maker(available_tensors)
    return new_dataset, number_of_channels

def train(encoder_name, epochs, validation_fraction, batch_size=32, pre_trained_weights=False):

    print("Getting train data...")
    train_data_path = osp.join(osp.dirname(data.__file__), "forest-biomass")
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    train_dataset, number_of_channels = prepare_dataset(patch_names, train_data_path)

    train_size = int(validation_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)

    base_model = smp.Unet(
        encoder_name=encoder_name,
        in_channels=number_of_channels,
        classes=1,
    )

    if pre_trained_weights:

        trained_weights_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights", f"{encoder_name}.pt")
        base_model.encoder.load_state_dict(torch.load(trained_weights_path))

    s2_model = Sentinel2Model(base_model)

    logger = TensorBoardLogger("tb_logs", name="SMP_model")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        logger=[logger],
    )
    # Train the model ⚡
    trainer.fit(s2_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    return s2_model

# def predict():
#
#     available = os.listdir(test_dataset)
#     chip_ids = set([x[0:8] for x in available])
#
#     test_image_paths = [os.path.join(test_dataset, x + "_S2_06.tif") for x in chip_ids]
#
#     for chip_id in tqdm(chip_ids):
#         image_path = os.path.join(test_dataset, f"{chip_id}_S2_06.tif")
#
#         test_im = torch.tensor(rasterio.open(image_path).read().astype(np.float32)[:10])
#         test_im = (test_im.permute(1, 2, 0) - test_im.mean(dim=(1, 2))) / (test_im.std(dim=(1, 2)) + 0.01)
#         test_im = test_im.permute(2, 0, 1)
#         pred = s2_model(test_im.unsqueeze(0))
#
#         im = Image.fromarray(pred.squeeze().cpu().detach().numpy())
#         im.save(f"preds/{chip_id}_agbm.tif", format="TIFF", save_all=True)
#
#     ## A quick visualization
#
#     # Load data
#     test_im = torch.tensor(rasterio.open("../train_features/29cc01ea_S2_06.tif").read().astype(np.float32)[:10])
#     test_label = torch.tensor(rasterio.open("../train_agbm/29cc01ea_agbm.tif").read().astype(np.float32)[:10])
#
#     # Normalize
#     test_im = (test_im.permute(1, 2, 0) - test_im.mean(dim=(1, 2))) / (test_im.std(dim=(1, 2)) + 0.01)
#     test_im = test_im.permute(2, 0, 1)
#
#     # Show ground truth
#     plt.imshow(test_label.permute(1, 2, 0).numpy(), interpolation='nearest')
#     plt.show()
#
#     # Predict
#     pred = s2_model(test_im.unsqueeze(0))
#
#     # Show predictions
#     plt.imshow(pred.cpu().squeeze().detach().numpy(), interpolation='nearest')
#     plt.show()

if __name__ == '__main__':
    train()
