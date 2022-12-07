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

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class SentinelDataset2(Dataset):
    def __init__(self, annotations_file, img_dir, label_dir=None, transform=None, target_transform=None):
        self.dataframe = pd.read_csv(annotations_file)
        self.chip_ids = self.dataframe.chip_id.unique()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform

        available = os.listdir(img_dir)

        self.chip_ids = [x for x in self.chip_ids if x + "_S2_06.tif" in available]

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.chip_ids[idx] + "_S2_06.tif")
        image = torch.tensor(rasterio.open(img_path).read().astype(np.float32)[:10])

        # Normalize image with mean 0 and stddev 1. Add a little bit to div to avoid dividing by 0
        image = (image.permute(1, 2, 0) - image.mean(dim=(1, 2))) / (image.std(dim=(1, 2)) + 0.01)
        image = image.permute(2, 0, 1)

        label_path = os.path.join(self.label_dir, self.chip_ids[idx] + "_agbm.tif")
        label = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, label


class SentinelDatasetAlternative(Dataset):
    def __init__(self, annotations_file, img_dir, label_dir=None, transform=None, target_transform=None):
        self.dataframe = pd.read_csv(annotations_file)
        self.chip_ids = self.dataframe.chip_id.unique()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.chip_ids = [img for img in os.listdir(img_dir) if "S2" in img]


    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        current_chip_id = self.chip_ids[idx][0:8]

        img_path = os.path.join(self.img_dir, self.chip_ids[idx])
        image = torch.tensor(rasterio.open(img_path).read().astype(np.float32)[:10])
        # image = torch.tensor(rasterio.open(img_path).read().astype(np.float32)[0])

        image = (image.permute(1, 2, 0) - image.mean(dim=(1, 2))) / (image.std(dim=(1, 2)) + 0.01)
        image = image.permute(2, 0, 1)

        label_path = os.path.join(self.label_dir, current_chip_id + "_agbm.tif")
        label = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        if self.transform:
            image = self.transform(image)

        print("###################################")

        print(len(image))
        print(image)

        print("-----------------------------------")

        print(len(label))
        print(label)

        print("###################################")

        sys.exit()

        return image, label


class Sentinel2Model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        # self.log("train/loss", loss)
        # self.log("train/rmse", torch.sqrt(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        # self.log("valid/loss", loss)
        # self.log("valid/rmse", torch.sqrt(loss))
        return loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.0001)]

    def forward(self, x):
        return self.model(x)


class Sentinel2ModelAlternative(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits[:, :, 0, 0]
        y = y[:, 0, 0, 0]
        y = y.long()
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits[:, :, 0, 0]
        y = y[:, 0, 0, 0]
        y = y.long()
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        print("Flag 1")
        # print(torch.optim.Adam(self.parameters(), lr=0.002))
        return torch.optim.Adam(self.parameters(), lr=0.002)

    def forward(self, image):
        return self.model(image)


def train():


    train_data_path = r'C:\Users\kuipe\OneDrive\Bureaublad\Epoch\forestbiomass\data\train_agbm'
    train_label_path = osp.abspath(osp.join(osp.realpath('__file__'), "../../../data/imgs/train_agbm"))
    metadata_path = osp.abspath(osp.join(osp.realpath('__file__'), "../../../data/features_metadata_FzP19JI.csv"))
    features_df = pd.read_csv(metadata_path)

    train_dataset = SentinelDatasetAlternative(metadata_path, train_data_path, label_dir=train_label_path)
    print(train_dataset[1].shape)

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6)
    valid_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=6)

    test_dataset = SentinelDatasetAlternative(metadata_path, train_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=6)

    base_model = smp.Unet(
        encoder_name="resnet50",
        in_channels=10,
        classes=1,
    )

    trained_weights_path = osp.abspath(
        osp.join(osp.realpath('__file__'), "../../../data/pre-trained_weights/resnet50.pt"))
    base_model.encoder.load_state_dict(torch.load(trained_weights_path))
    s2_model = Sentinel2Model(base_model)

    logger = TensorBoardLogger("../../data/tb_logs", name="Unet_resnet50")

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=5,
        logger=[logger],
    )
    # Train the model ⚡
    trainer.fit(s2_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


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
