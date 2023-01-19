# 12 x (4 + 11) bands in
# missing month = zeros
#
# input = 180 bands
# output = 180 values between 0 - 1
# 	- indicating %corrupted, fully corrupted = 1
# 	- missing = 1

import torch
from torch.nn import functional as F


def create_mask(image_tensor, radius):
    assert image_tensor.shape == (256, 256), "Input image must be of shape (256, 256)"
    # add singleton dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Create convolution mask
    convolution_mask = F.avg_pool2d(image_tensor, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    # remove singleton dimension
    convolution_mask = convolution_mask.squeeze(0)
    # Create comparison mask
    comparison_mask = torch.where(image_tensor == convolution_mask, torch.tensor(0.), torch.tensor(1.))
    return comparison_mask[0]


def extract_metadata_band(band, missing):
    if missing:
        return 0
    else:
        return torch.mean(create_mask(band, radius=2))


def extract_metadata_batch(batch):
    batch_metadata = []
    for patch in batch:

        # TODO: see how missing bands are treated in the dataloader we want to use and change code accordingly
        patch_metadata = []
        for band in patch:
            patch_metadata.append(extract_metadata_band(band, False))

        batch_metadata.append(patch_metadata)
    return torch.FloatTensor(batch_metadata)
