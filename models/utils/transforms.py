import numpy as np
import torch
import torch.nn as nn

from models.utils.check_corrupted import is_corrupted, is_cloud

_EPSILON = 1e-10


class AGBMLog1PScale(nn.Module):
    """Apply ln(x + 1) Scale to AGBM Target Data"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        inputs["label"] = torch.log1p(inputs["label"])
        return inputs


class ClampAGBM(nn.Module):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    def __init__(self, vmin=0.0, vmax=500.0) -> None:
        """Initialize ClampAGBM
        Args:
            vmin (float): minimum clamp value
            vmax (float): maximum clamp value, 500 is reasonable default per empirical analysis of AGBM data
        """
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, inputs):
        inputs["label"] = torch.clamp(inputs["label"], min=self.vmin, max=self.vmax)
        return inputs


class DropBands(nn.Module):
    """Drop specified bands by index"""

    def __init__(self, device, bands_to_keep=None) -> None:
        super().__init__()
        self.device = device
        self.bands_to_keep = bands_to_keep

    def forward(self, inputs):
        if not self.bands_to_keep:
            return inputs
        X = inputs["image"].detach()
        if X.ndim == 4:
            slice_dim = 1
        else:
            slice_dim = 0
        inputs["image"] = X.index_select(
            slice_dim, torch.tensor(self.bands_to_keep, device=self.device)
        )
        return inputs


class Sentinel2Scale(nn.Module):
    """Scale Sentinel 2 optical channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        scale_val = (
            4000.0  # True scaling is [0, 10000], most info is in [0, 4000] range
        )
        X = X / scale_val

        # CLP values in band 10 are scaled differently than optical bands, [0, 100]
        if X.ndim == 4:
            X[:][10] = X[:][10] * scale_val / 100.0
        else:
            X[10] = X[10] * scale_val / 100.0
        return X.clamp(0, 1.0)


class Sentinel1Scale(nn.Module):
    """Scale Sentinel 1 SAR channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        s1_max = (
            20.0  # S1 db values range mostly from -50 to +20 per empirical analysis
        )
        s1_min = -50.0
        X = (X - s1_min) / (s1_max - s1_min)
        return X.clamp(0, 1)


class AppendRatioAB(nn.Module):
    """Append the ratio of specified bands to the tensor."""

    def __init__(self, index_a, index_b):
        """Initialize a new transform instance.
        Args:
            index_a: numerator band channel index
            index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b

    def _compute_ratio(self, band_a, band_b):
        """Compute ratio band_a/band_b.
        Args:
            band_a: numerator band tensor
            band_b: denominator band tensor
        Returns:
            band_a/band_b
        """
        return band_a / (band_b + _EPSILON)

    def forward(self, sample):
        """Compute and append ratio to input tensor.
        Args:
            sample: dict with tensor stored in sample['image']
        Returns:
            the transformed sample
        """
        X = sample["image"].detach()
        ratio = self._compute_ratio(
            band_a=X[..., self.index_a, :, :],
            band_b=X[..., self.index_b, :, :],
        )
        ratio = ratio.unsqueeze(self.dim)
        sample["image"] = torch.cat([X, ratio], dim=self.dim)
        return sample


def select_transform_method(picked_transform_method, in_channels, image_size=256):
    if picked_transform_method == "replace_corrupted_0s":
        # return replace_corrupted_0s, 0
        return ReplaceCorruptedZeros(image_size), 0
    elif picked_transform_method == "replace_corrupted_noise":
        return ReplaceCorruptedNoise(), 0
    elif picked_transform_method == "add_band_corrupted_arrays":
        # return add_band_corrupted_arrays, in_channels
        return AppendZeros(), in_channels
    else:
        # return no_transformation, 0
        return NoTransformation(), 0


class NoTransformation(nn.Module):
    """No transformation"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return X


class ReplaceCorruptedZeros(nn.Module):
    """Replace corrupted bands by zeros"""

    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = image_size

    def forward(self, inputs):
        X = inputs["image"].detach()
        for b, x in enumerate(X):
            for ind, band in enumerate(x):
                if is_corrupted(np.array(band), self.image_size):
                    X[b, ind] = torch.zeros((self.image_size, self.image_size))
        inputs["image"] = X
        return inputs


class ReplaceClouds(nn.Module):
    """Replace clouds by zeros"""

    def __init__(self, cloud_channel=10):
        super().__init__()
        self.cloud_channel = cloud_channel

    def forward(self, inputs):
        X = inputs["image"].detach()
        for b, x in enumerate(X):
            if is_cloud(np.array(x[self.cloud_channel, :, :])):
                X[b] = torch.zeros(x.shape)
        inputs["image"] = X
        return inputs


class ReplaceCorruptedNoise(nn.Module):
    """Replace corrupted bands by noise"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        X = inputs["image"].detach()

        for ind, band in enumerate(X):
            if is_corrupted(np.array(band)):
                X[ind] = torch.rand((256, 256))
        inputs["image"] = X
        return inputs


class AppendZeros(nn.Module):
    """Append tensors of 0s or 1s depending on corruption"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        X = inputs["image"].detach()

        for ind, band in enumerate(X):
            if is_corrupted(np.array(band)):
                X = torch.cat((X, torch.zeros((1, 256, 256))), 0)
            else:
                X = torch.cat((X, torch.ones((1, 256, 256))), 0)
        inputs["image"] = X
        return inputs
