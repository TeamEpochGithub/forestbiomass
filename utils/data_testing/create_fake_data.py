import os
import numpy as np
import random

def gen_fake_patch_id():
    return "".join([random.choice('ABCDEFGHIJKLMNPQRSTUVWXYZ012345678') for _ in range(10)])


for _ in range(32):
    patch = gen_fake_patch_id()

    os.mkdir(f"../../data/fake_data/{patch}")

    file_name_label = f"../../data/fake_data/{patch}/label.npy"
    fake_label = np.ones((256, 256))

    np.save(file_name_label, fake_label)

    # Creating fake data while merging the s1 and s2 bands together so 4+11 = 15
    # for month in range(12):
    #     file_name = f"../data/imgs/fake_data/{patch}/{patch}_{month}.npy"
    #     file_name_label = f"../data/imgs/fake_data/{patch}/label.npy"
    #     fake_array = np.ones((15, 256, 256)) * random.randint(0, 255)
    #     fake_label = np.ones((256, 256)) * random.randint(0, 255)
    #     np.save(file_name, fake_array)
    #     np.save(file_name_label, fake_label)

    #Creating separate arrays for both s1 and s2
    for month in range(12):
        os.mkdir(f"../../data/fake_data/{patch}/{month}")
        os.mkdir(f"../../data/fake_data/{patch}/{month}/S1")
        os.mkdir(f"../../data/fake_data/{patch}/{month}/S2")

        for band in range(4):
            file_name_s1 = f"../../data/fake_data/{patch}/{month}/S1/{band}.npy"
            fake_array_s1 = np.ones((256, 256))
            np.save(file_name_s1, fake_array_s1)

        for band in range(11):
            file_name_s2 = f"../../data/fake_data/{patch}/{month}/S2/{band}.npy"
            fake_array_s2 = np.ones((256, 256))
            np.save(file_name_s2, fake_array_s2)


