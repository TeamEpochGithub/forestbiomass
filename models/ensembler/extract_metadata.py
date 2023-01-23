# 12 x (4 + 11) bands in
# missing month = zeros
#
# input = 180 bands
# output = 180 values between 0 - 1
# 	- indicating %corrupted, fully corrupted = 1
# 	- missing = 1

import torch


def extract_metadata_band(band, missing):
    if missing:
        return 1
    else:
        # return the percentage of pixels that are the same as half of their closest 50 neighbours
        # for edge pixels the threshold should be increased according to how close they are to the edge
        pass


def extract_metadata_batch(batch):
    batch_metadata = []
    for patch in batch:

        # TODO: see how missing bands are treated in the dataloader we want to use and change code accordingly
        patch_metadata = []
        for band in patch:
            patch_metadata.append(extract_metadata_band(band, False))

        batch_metadata.append(patch_metadata)
    return torch.FloatTensor(batch_metadata)
