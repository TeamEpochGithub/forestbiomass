import torch
from PIL import Image
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import os
import rasterio
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import warnings
import numpy as np
import os.path as osp
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

import models
import data
import csv
from models.utils import loss_functions
import argparse
import models.utils.transforms as tf
import torch.nn.functional as F
from models.utils.patched_dataloading import (
    SentinelDataLoader,
    create_tensor,
    SubmissionDataLoader,
    apply_transforms,
    divide_into_patches,
    reasamble_patches,
)
import operator
import sys
import wandb
from pytorch_lightning.loggers import WandbLogger
from models.utils.simple_tensor_accumulate import accumulate_predictions

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Sentinel2Model(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.learning_rate = args.learning_rate
        self.train_loss_function = args.train_loss_function
        self.val_loss_function = args.val_loss_function

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.train_loss_function(y_hat, y)
        self.log("train/loss", loss, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.val_loss_function(y_hat, y)
        self.log("val/loss", loss, rank_zero_only=True)
        return loss

    def on_validation_epoch_end(self):
        return None

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        return self.model(x)


class PatchedSentinel2Model(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.learning_rate = args.learning_rate
        self.train_loss_function = args.train_loss_function
        self.val_loss_function = args.val_loss_function
        self.image_patch_size = args.image_patch_size
        self.num_patches = (256 // self.image_patch_size) ** 2  # symmetrical
        self.num_patches_per_dim = int(np.sqrt(self.num_patches))
        self.args = args
        # self.norm_map = norm_map

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(x.shape, y.shape)
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        y = y.view(-1, y.shape[-3], y.shape[-2], y.shape[-1])
        y_hat = self.model(x)
        loss = self.train_loss_function(y_hat, y)
        self.log("train/loss", loss, rank_zero_only=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        y = y.view(-1, y.shape[2], y.shape[3], y.shape[4])
        norm = F.unfold(
            torch.ones(
                (
                    self.args.batch_size,
                    1,
                    self.args.image_size,
                    self.args.image_size,
                )
            ),
            kernel_size=self.args.image_patch_size,
            stride=self.args.stride,
        )
        norm_map = F.fold(
            norm,
            (self.args.image_size, self.args.image_size),
            kernel_size=self.args.image_patch_size,
            stride=self.args.stride,
        )
        y_hat = self.model(x)
        loss = self.val_loss_function(y_hat, y)
        self.log("val/loss", loss, rank_zero_only=True)
        y_hat_orig = reasamble_patches(
            y_hat,
            self.args.image_patch_size,
            self.args.stride,
            self.args.batch_size,
            norm_map,
        )
        y_orig = reasamble_patches(
            y,
            self.args.image_patch_size,
            self.args.stride,
            self.args.batch_size,
            norm_map,
        )
        orig_loss = self.val_loss_function(y_hat_orig, y_orig)
        self.log("val/full_loss", orig_loss, rank_zero_only=True)
        return loss

    def on_validation_epoch_end(self):
        return None

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        return self.model(x)


def prepare_dataset(args):
    with open(args.training_ids_path, newline="\n") as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]
    # chip_ids = chip_ids[0:500]

    id_month_list = []
    for id in chip_ids:
        for month in range(0, 12):
            month_patch_path = osp.join(
                args.converted_training_features_path, id, str(month)
            )
            if osp.exists(osp.join(month_patch_path, "S2")):
                id_month_list.append((id, str(month)))

    in_channels = len(args.bands_to_keep)
    corrupted_transform_method, transform_channels = tf.select_transform_method(
        args.transform_method,
        in_channels=in_channels,
        image_size=args.image_patch_size,
    )
    # print(len(id_month_list))
    dataset = SentinelDataLoader(args, id_month_list, corrupted_transform_method)
    # print(len(dataset))
    return dataset, (in_channels + transform_channels)


def select_segmenter(encoder_weights, segmenter_name, encoder_name, number_of_channels):
    if segmenter_name == "Unet":
        base_model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=number_of_channels,
            classes=1,
            encoder_weights=encoder_weights,
        )
    else:
        base_model = None

    assert base_model is not None, "Segmenter name was not recognized."
    return base_model


def train(args):
    print("=" * 30)
    for arg in vars(args):
        print("--", arg, ":", getattr(args, arg))
    print("=" * 30)
    print("Getting train data...")

    train_dataset, number_of_channels = prepare_dataset(args)
    # print(len(train_dataset))
    # train_size = int(
    #     (1 - args.validation_fraction)
    #     * (len(train_dataset) // train_dataset.num_patches)
    # )
    # valid_size = len(train_dataset) // train_dataset.num_patches - train_size
    # train_size, valid_size = (
    #     train_dataset.num_patches * train_size,
    #     train_dataset.num_patches * valid_size,
    # )
    # print(train_size, valid_size)
    train_size = int(1 - args.validation_fraction * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_workers,
    )
    valid_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
    )
    base_model = select_segmenter(
        args.encoder_weights, args.segmenter_name, args.encoder_name, number_of_channels
    )

    pre_trained_weights_dir_path = osp.join(
        osp.dirname(data.__file__), "pre-trained_weights"
    )

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(
            pre_trained_weights_dir_path, f"{args.encoder_name}.pt"
        )
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    if args.patched:
        model = PatchedSentinel2Model(base_model, args)
    else:
        model = Sentinel2Model(base_model, args)

    logger = TensorBoardLogger(
        osp.join(args.models_path, "tb_logs"),
        name=args.model_identifier,
    )
    wandb_logger = WandbLogger(
        name=args.model_identifier,
        project="forestbiomass",
        entity="team-epoch",
        tags="",
        log_model=False,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.save_top_k_checkpoints,
        monitor="val/loss",
        mode="min",
    )

    # ddp = DDPStrategy(process_group_backend="gloo")
    # wandb_logger.watch(model, log="all", log_graph=False)
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=[logger, wandb_logger],
        log_every_n_steps=args.log_step_frequency,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        precision=args.precision,
        accelerator="cpu",
        # devices=[0],
        # num_nodes=4,
        #  strategy="ddp",
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )
    wandb.finish()

    return model, str(trainer.callback_metrics["val/loss"].item())


def load_model(args):
    print("Getting saved model...")

    assert osp.exists(args.current_model_path) is True, "requested model does not exist"
    log_folder_path = args.current_model_path

    version_dir = list(os.scandir(log_folder_path))[args.model_version]

    checkpoint_dir_path = osp.join(log_folder_path, version_dir, "checkpoints")
    latest_checkpoint_name = list(os.scandir(checkpoint_dir_path))[-1]
    latest_checkpoint_path = osp.join(checkpoint_dir_path, latest_checkpoint_name)

    base_model = select_segmenter(
        args.encoder_weights,
        args.segmenter_name,
        args.encoder_name,
        len(args.bands_to_keep) + args.extra_channels,
    )

    # This block might be redundant if we can download weights via the python segmentation models library.
    # However, it might be that not all weights are available this way.
    # If you have downloaded weights (in the .pt format), put them in the pre-trained-weights folder
    # and give the file the same name as the encoder you're using.
    # If you do that, this block will try and load them for your model.
    pre_trained_weights_dir_path = osp.join(
        osp.dirname(data.__file__), "pre-trained_weights"
    )

    if osp.exists(osp.join(pre_trained_weights_dir_path, f"{args.encoder_name}.pt")):
        pre_trained_weights_path = osp.join(
            pre_trained_weights_dir_path, f"{args.encoder_name}.pt"
        )
    else:
        pre_trained_weights_path = None

    if pre_trained_weights_path is not None:
        base_model.encoder.load_state_dict(torch.load(pre_trained_weights_path))

    ###########################################################

    if args.patched:
        model = PatchedSentinel2Model(base_model, args)
    else:
        model = Sentinel2Model(base_model, args)

    checkpoint = torch.load(str(latest_checkpoint_path))
    model.load_state_dict(checkpoint["state_dict"])

    print("Model loaded")
    return model


def create_predictions(args):
    model = load_model(args)
    test_data_path = args.testing_features_path

    with open(args.testing_ids_path, newline="") as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    predictions = []
    for index, id in enumerate(patch_names):

        all_months = []
        for month in range(0, 12):
            s1_folder_path = osp.join(test_data_path, id, f"{month:02}", "S1")
            s2_folder_path = osp.join(test_data_path, id, f"{month:02}", "S2")

            if osp.exists(s2_folder_path):
                all_bands = []
                for s1_index in range(0, 4):
                    if args.S1_band_selection[s1_index] == 1:
                        band = np.load(
                            osp.join(s1_folder_path, f"{s1_index}.npy"),
                            allow_pickle=True,
                        )
                        all_bands.append(band)

                for s2_index in range(0, 10):
                    if args.S2_band_selection[s2_index] == 1:
                        band = np.load(
                            osp.join(s2_folder_path, f"{s2_index}.npy"),
                            allow_pickle=True,
                        )
                        all_bands.append(band)

                input_tensor = create_tensor(all_bands)
                pred = model(input_tensor.unsqueeze(0))
                pred = pred.cpu().squeeze().detach().numpy()
                all_months.append(pred)

        count = len(all_months)
        prediction = np.asarray(sum(all_months) / count)
        predictions.append(prediction)

        total = len(patch_names)
        if index % 100 == 0:
            print(f"{index} / {total}")

    return predictions


def create_submissions(args):
    model = load_model(args)
    test_data_path = args.converted_testing_features_path

    with open(args.testing_ids_path, newline="") as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    predictions = []
    for index, id in tqdm(enumerate(patch_names)):

        all_months = []
        for month in range(0, 12):

            s1_folder_path = osp.join(test_data_path, id, f"{month:02}", "S1")
            s2_folder_path = osp.join(test_data_path, id, f"{month:02}", "S2")

            if osp.exists(s2_folder_path):
                all_list = []

                for s1_index in range(11):
                    band = np.load(
                        osp.join(s2_folder_path, f"{s1_index}.npy"), allow_pickle=True
                    )
                    all_list.append(band)

                for s2_index in range(4):
                    band = np.load(
                        osp.join(s1_folder_path, f"{s2_index}.npy"), allow_pickle=True
                    )
                    all_list.append(band)

                all_tensor = create_tensor(all_list)
                label_tensor = torch.rand((1, 256, 256))
                all_tensor = divide_into_patches(all_tensor, args.image_patch_size)
                label_tensor = divide_into_patches(label_tensor, args.image_patch_size)
                print(all_tensor.shape)
                sample = {
                    "image": all_tensor,
                    "label": label_tensor,
                }  # 'image' and 'label' are used by torchgeo

                (
                    corrupted_transform_method,
                    transform_channels,
                ) = tf.select_transform_method(
                    args.transform_method, in_channels=len(args.bands_to_keep)
                )

                selected_tensor = apply_transforms(
                    bands_to_keep=args.bands_to_keep,
                    corrupted_transform_method=corrupted_transform_method,
                )(sample)

                selected_tensor = selected_tensor["image"]
                pred = model(selected_tensor.unsqueeze(0))
                norm = F.unfold(
                    torch.ones(pred.shape),
                    kernel_size=args.image_patch_size,
                    stride=args.stride,
                )
                norm_map = F.fold(
                    norm,
                    pred.shape[-2:],
                    kernel_size=args.image_patch_size,
                    stride=args.stride,
                )
                pred = reasamble_patches(
                    pred,
                    args.image_patch_size,
                    args.stride,
                    args.batch_size,
                    norm_map,
                )
                pred = pred.cpu().squeeze().detach().numpy()
                all_months.append(pred)

        count = len(all_months)

        if count == 0:
            continue

        agbm_arr = np.asarray(sum(all_months) / count)

        test_agbm_path = osp.join(args.submission_folder_path, f"{id}_agbm.tif")
        im = Image.fromarray(agbm_arr)
        im.save(test_agbm_path)

        total = len(patch_names)

        if index % 100 == 0:
            print(f"{index} / {total}")

    return predictions


def experimental_submission(args):
    model = load_model(args)

    with open(args.testing_ids_path, newline="") as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    chip_ids = patch_name_data[0]

    id_month_list = []

    if args.data_type == "npy":

        for id in chip_ids:
            for month in range(0, 12):
                month_patch_path = osp.join(
                    args.converted_testing_features_path, id, str(month)
                )
                if osp.exists(osp.join(month_patch_path, "S2")):
                    id_month_list.append((id, str(month)))

        corrupted_transform_method, transform_channels = tf.select_transform_method(
            args.transform_method, in_channels=len(args.bands_to_keep)
        )
        new_dataset = SubmissionDataLoader(
            args, id_month_list, corrupted_transform_method
        )
    else:
        sys.exit("Error: Invalid data type selected.")

    trainer = Trainer(accelerator="cpu", devices=1)
    dl = DataLoader(
        new_dataset,
        num_workers=4,
        batch_size=1,  # args.batch_size,
        shuffle=False,
    )

    predictions = trainer.predict(model, dataloaders=dl)
    predictions = [
        reasamble_patches(
            prediction,
            args.image_patch_size,
            args.stride,
            args.batch_size,
            norm_map,
        )
        for prediction in predictions
    ]
    tensor_id_list = [i[0] for i in id_month_list]

    transformed_predictions = [x.cpu().squeeze().detach().numpy() for x in predictions]
    linked_tensor_list = list(zip(tensor_id_list, transformed_predictions))
    linked_tensor_list = sorted(linked_tensor_list, key=operator.itemgetter(0))

    averaged_tensor_list = list(accumulate_predictions(linked_tensor_list))

    for id_tensor_pair in averaged_tensor_list:
        current_id = id_tensor_pair[0]
        current_tensor = id_tensor_pair[1]

        agbm_path = osp.join(args.submission_folder_path, f"{current_id}_agbm.tif")

        im = Image.fromarray(current_tensor)
        im.save(agbm_path)

    print("Finished creating submission.")


def set_args():
    model_segmenter = "Unet"
    model_encoder = "efficientnet-b2"
    model_encoder_weights = "imagenet"  # Leave None if not using weights.
    data_type = "npy"  # options are "npy" or "tiff"
    epochs = 100
    learning_rate = 1e-4
    dataloader_workers = 4
    validation_fraction = 0.2
    batch_size = 1
    log_step_frequency = 10
    version = -1  # Keep -1 if loading the latest model version.
    save_top_k_checkpoints = 1
    transform_method = "replace_corrupted_0s"  # "replace_corrupted_noise"  # nothing  # add_band_corrupted_arrays
    train_loss_function = loss_functions.rmse_loss
    val_loss_function = loss_functions.rmse_loss
    patched = True

    # WARNING: Only increment extra_channels when making predictions/submission (based on the transform method used)
    # it is automatically incremented during training based on the transform method used (extra channels generated)
    extra_channels = 0

    band_map = {
        # S2 bands
        0: "S2-B2: Blue-10m",
        1: "S2-B3: Green-10m",
        2: "S2-B4: Red-10m",
        3: "S2-B5: VegRed-704nm-20m",
        4: "S2-B6: VegRed-740nm-20m",
        5: "S2-B7: VegRed-780nm-20m",
        6: "S2-B8: NIR-833nm-10m",
        7: "S2-B8A: NarrowNIR-864nm-20m",
        8: "S2-B11: SWIR-1610nm-20m",
        9: "S2-B12: SWIR-2200nm-20m",
        10: "S2-CLP: CloudProb-160m",
        # S1 bands
        11: "S1-VV-Asc: Cband-10m",
        12: "S1-VH-Asc: Cband-10m",
        13: "S1-VV-Desc: Cband-10m",
        14: "S1-VH-Desc: Cband-10m",
        # Bands derived by transforms
        15: "S2-NDVI: (NIR-Red)/(NIR+Red) 10m",
        16: "S1-NDVVVH-Asc: Norm Diff VV & VH, 10m",
        17: "S2-NDBI: Difference Built-up Index, 20m",
        18: "S2-NDRE: Red Edge Vegetation Index, 20m",
        19: "S2-NDSI: Snow Index, 20m",
        20: "S2-NDWI: Water Index, 10m",
        21: "S2-SWI: Sandardized Water-Level Index, 20m",
        22: "S1-VV/VH-Asc: Cband-10m",
        23: "S2-VV/VH-Desc: Cband-10m",
    }

    bands_to_keep = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    band_indicator = ["1" if k in bands_to_keep else "0" for k, v in band_map.items()]
    image_patch_size = 32
    stride = 8
    image_size = 256
    precision = 16

    parser = argparse.ArgumentParser()
    bands_to_keep_indicator = "bands-" + "".join(str(x) for x in band_indicator)
    model_identifier = f"{model_segmenter}_{model_encoder}_{bands_to_keep_indicator}_lr-{learning_rate}_bs-{batch_size}_transform-{transform_method}_loss-{train_loss_function.__name__}_precision-{precision}_patched-{patched}"

    parser.add_argument("--model_identifier", default=model_identifier, type=str)
    parser.add_argument("--segmenter_name", default=model_segmenter, type=str)
    parser.add_argument("--encoder_name", default=model_encoder, type=str)
    parser.add_argument("--encoder_weights", default=model_encoder_weights, type=str)
    parser.add_argument("--model_version", default=version, type=int)
    parser.add_argument("--data_type", default=data_type, type=str)

    data_path = osp.dirname(data.__file__)
    models_path = osp.dirname(models.__file__)
    parser.add_argument(
        "--models_path",
        default=str(models_path),
        type=str,
    )
    parser.add_argument(
        "--converted_training_features_path",
        default=str(osp.join(data_path, "converted")),
        type=str,
    )
    parser.add_argument(
        "--converted_testing_features_path",
        default=str(osp.join(data_path, "testing_converted")),
        type=str,
    )

    parser.add_argument(
        "--training_ids_path", default=str(osp.join(data_path, "patch_names")), type=str
    )
    parser.add_argument(
        "--testing_ids_path",
        default=str(osp.join(data_path, "test_patch_names")),
        type=str,
    )
    parser.add_argument(
        "--current_model_path",
        default=str(osp.join(models_path, "tb_logs", model_identifier)),
        type=str,
    )
    parser.add_argument(
        "--submission_folder_path",
        default=str(osp.join(data_path, "imgs", "test_agbm")),
        type=str,
    )

    parser.add_argument("--dataloader_workers", default=dataloader_workers, type=int)
    parser.add_argument("--batch_size", default=batch_size, type=int)
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--learning_rate", default=learning_rate, type=float)
    parser.add_argument(
        "--validation_fraction", default=validation_fraction, type=float
    )
    parser.add_argument("--log_step_frequency", default=log_step_frequency, type=int)
    parser.add_argument(
        "--save_top_k_checkpoints", default=save_top_k_checkpoints, type=int
    )

    parser.add_argument("--bands_to_keep", default=bands_to_keep, type=list)
    parser.add_argument("--train_loss_function", default=train_loss_function)
    parser.add_argument("--val_loss_function", default=val_loss_function)
    parser.add_argument("--transform_method", default=transform_method, type=str)
    parser.add_argument("--extra_channels", default=extra_channels, type=int)
    parser.add_argument(
        "--image_patch_size",
        default=image_patch_size,
        type=int,
        help="target size for the divided (PatchedUp) image",
    )
    parser.add_argument("--stride", default=stride, type=int)
    parser.add_argument("--image_size", default=image_size, type=int)
    parser.add_argument("--patched", default=patched, type=bool)
    parser.add_argument("--precision", default=precision, type=int)

    args = parser.parse_args()
    # assert (
    #     args.batch_size // (256 / args.image_patch_size) ** 2
    #     == args.batch_size / (256 / args.image_patch_size) ** 2
    # ), "Batch size must be divisible by image patch size"

    return args


if __name__ == "__main__":
    args = set_args()
    _, score = train(args)
    # print(score)

    # create_submissions(args)
    # experimental_submission(args)
