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
    """
    @param batch: A torch tensor of shape (batch_size, 12 * num_bands, 256, 256)
    @return: A torch tensor of shape (batch_size, 12 * num_bands) indicating "corruptedness" for each band, where 0 is very corrupted
    and 1 is not corrupted
    """
    batch_metadata = []
    for patch in batch:

        patch_metadata = []
        for band in patch:
            patch_metadata.append(extract_metadata_band(band, False))

        batch_metadata.append(patch_metadata)
    return torch.FloatTensor(batch_metadata)

if __name__ == '__main__':
    # very_corrupted = torch.zeros(16, 180, 256, 256)
    # print(extract_metadata_batch(very_corrupted))

    non_corrupted = torch.rand(16, 180, 256, 256)
    print(extract_metadata_batch(non_corrupted).shape)

