import sys

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import os
import rasterio
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import warnings
import numpy as np
import os.path as osp
from pytorch_lightning.loggers import TensorBoardLogger
import data
import csv

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

        # print("###################################")
        #
        # print(len(data_tensor))
        # print(data_tensor)
        #
        # print("-----------------------------------")
        #
        # print(len(label_tensor))
        # print(label_tensor)
        #
        # print("###################################")
        #
        # sys.exit()
        #
        # sys.exit()

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

        label_tensor = torch.tensor(np.asarray([label], dtype=np.float32))

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

                data_tensor = torch.tensor(np.asarray(arr_list, dtype=np.float32))

                # These operations were present in the original notebook. I'll figure out some other time what they do.
                # By some other time I mean never.
                data_tensor = (data_tensor.permute(1, 2, 0) - data_tensor.mean(dim=(1, 2))) / (data_tensor.std(dim=(1, 2)) + 0.01)
                data_tensor = data_tensor.permute(2, 0, 1)

                available_tensors.append((data_tensor, label_tensor))

            except IOError as e:
                continue

    new_dataset = Segmentation_Dataset_Maker(available_tensors)
    return new_dataset, number_of_channels


def select_segmenter(segmenter_name, encoder_name, number_of_channels):

    if segmenter_name == "Unet":

        base_model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=number_of_channels,
            classes=1
        )

    else:
        base_model = None

    assert base_model is not None, "Segmenter name was not recognized."

    return base_model


def create_tensor(band_list):

    band_list = np.asarray(band_list, dtype=np.float32)

    band_tensor = torch.tensor(band_list)
    band_tensor = (band_tensor.permute(1, 2, 0) - band_tensor.mean(dim=(1, 2))) / (band_tensor.std(dim=(1, 2)) + 0.01)
    band_tensor = band_tensor.permute(2, 0, 1)

    return band_tensor.unsqueeze(0)


def train(segmenter_name, encoder_name, epochs, training_fraction, batch_size=32):

    print("Getting train data...")
    train_data_path = osp.join(osp.dirname(data.__file__), "forest-biomass")
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    train_dataset, number_of_channels = prepare_dataset(patch_names, train_data_path)

    train_size = int(training_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)
    valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)

    base_model = select_segmenter(segmenter_name, encoder_name, number_of_channels)

    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    s2_model = Sentinel2Model(base_model)

    logger = TensorBoardLogger("tb_logs", name=f"{segmenter_name}_{encoder_name}")

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        logger=[logger],
        log_every_n_steps=5
    )
    # Train the model ⚡
    trainer.fit(s2_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    return s2_model

def load_model(segmenter_name, encoder_name, number_of_channels, version=None):

    print("Getting saved model...")
    log_folder_path = osp.join(osp.dirname(data.__file__), "tb_logs", f"{segmenter_name}_{encoder_name}")

    if version is None:
        version_dir = list(os.scandir(log_folder_path))[-1]
    else:
        version_dir = list(os.scandir(log_folder_path))[version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")

    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]

    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    base_model = select_segmenter(segmenter_name, encoder_name, number_of_channels)

    pre_trained_weights_dir_path = osp.join(osp.dirname(data.__file__), "pre-trained_weights")

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{encoder_name}.pt")):
        pre_trained_weights_path = osp.join(pre_trained_weights_dir_path, f"{encoder_name}.pt")
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    s2_model = Sentinel2Model(base_model)

    checkpoint = torch.load(latest_checkpoint_path)
    s2_model.load_state_dict(checkpoint["state_dict"])

    return s2_model

if __name__ == '__main__':
    model = load_model("Unet", "resnet50", 10)

    example_path = osp.join(osp.dirname(data.__file__), "forest-biomass", "2d977c85", "0", "S2")

    collected_bands = []

    for i in range(0, 10):
        collected_bands.append(np.load(osp.join(example_path, f"{i}.npy"), allow_pickle=True))

    model_input = create_tensor(collected_bands)

    prediction = model(model_input)

    plt.imshow(prediction.cpu().squeeze().detach().numpy(), interpolation='nearest')
    plt.show()
