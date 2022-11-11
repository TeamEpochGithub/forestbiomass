import os
import numpy as np
import random

def gen_fake_patch_id():
    return "".join([random.choice('ABCDEFGHIJKLMNPQRSTUVWXYZ012345678') for _ in range(10)])


for _ in range(64):
    patch = gen_fake_patch_id()

    os.mkdir(f"../data/imgs/fake_data/{patch}")

    for month in range(12):
        file_name = f"../data/imgs/fake_data/{patch}/{patch}_{month}.npy"
        file_name_label = f"../data/imgs/fake_data/{patch}/label.npy"
        fake_array = np.ones((15, 256, 256)) * random.randint(0, 255)
        fake_label = np.ones((256, 256)) * random.randint(0, 255)
        np.save(file_name, fake_array)
        np.save(file_name_label, fake_label)