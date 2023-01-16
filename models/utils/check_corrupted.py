from collections import Counter
import numpy as np


def is_corrupted(band, image_size=256, samples_per_dim=4):
    """
    Returns if band is corrupted by checking if half of evenly sampled pixels are identical (naive)
    """
    xl = np.linspace(0, image_size - 1, samples_per_dim)
    yl = np.linspace(0, image_size - 1, samples_per_dim)
    # xv, yv = np.meshgrid(x, y)
    samples = []
    for x in xl:
        for y in yl:
            x = x.astype(int)
            y = y.astype(int)
            samples.append(band[x, y])
    c = Counter(samples)
    return c.most_common(1)[0][1] > (samples_per_dim**2) / 2


def is_cloud(clp):
    clp /= 255
    return np.mean(clp) > 0.5
