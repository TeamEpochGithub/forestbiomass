#Imports
import tensorflow as tf
from keras import datasets, layers, models

import numpy as np

import os
from os import path

# from classification_models.tfkeras import Classifiers

np.random.seed(0)

data_path = "../../data/fake_data/"
# train_features_path = "../../data/imgs/train_features"
# train_abgm_path = "../../data/imgs/fake_data2/"

class LocalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=16):
        """
        Few things to mention:
            - The data generator tells our corrupted_model how to fetch one batch of training data (in this case from files)
            - Any work that can be done before training, should be done in init, since we want fetching a batch to be fast
            - Therefore, we want all filenames and labels to be determined before training
            - This saves work, because we will be fetching batches multiple times (across epochs)
        """
        # Get all filenames in directory
        self.filenames = [dir_path + file for file in os.listdir(dir_path)]
        print(self.filenames)
        print(len(self.filenames))
        # Include batch size as attribute
        self.batch_size = batch_size

    def __len__(self):
        """
        Should return the number of BATCHES the generator can retrieve (partial batch at end counts as well)
        """
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Tells generator how to retrieve BATCH idx
        """
        # Get filenames for X batch
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        label_blobs = []
        for file in batch_filenames:
            label = np.load(file + "/label.npy")
            if not label.any():
                batch_filenames.remove(file)
                continue
            label_blobs.append(label)

        batch_y = np.stack([f for f in label_blobs], axis=0)

        # concatenating all the months together and then stacking on each other for both s1 and s2 separate
        # List for all the patches together, to be processed to batch
        batch_s1_array = []
        batch_s2_array = []

        missing_s2 = []
        for file in batch_filenames:
            s1_data = []
            s2_data = []
            missing_s2_patch = []
            for month in range(12):
                missing = 0
                s1_temp = []
                for band in range(4):
                    s1_temp.append(np.load(file + '/' + str(month) + "/S1/" + str(band) + ".npy"))
                s1_data.append(s1_temp)

                s2_temp = []
                if path.exists(file + '/' + str(month) + "/S2/"):
                    for band in range(11):
                        s2_temp.append(np.load(file + '/' + str(month) + "/S2/" + str(band) + ".npy"))
                else:
                    missing = 1
                    for band in range(11):
                        fake_array_s2 = np.zeros((256, 256), dtype='uint16')
                        s2_temp.append(fake_array_s2)
                s2_data.append(s2_temp)
                missing_s2_patch.append(missing)

            batch_s1_array.append(np.concatenate(s1_data))
            batch_s2_array.append(np.concatenate(s2_data))

            missing_s2.append(missing_s2_patch)

        batch_x_s1 = np.stack([bs1 for bs1 in batch_s1_array])
        batch_x_s2 = np.stack([bs2 for bs2 in batch_s2_array])

        # try:
        #     batch_x_s1 = np.stack([np.array(np.concatenate([np.load(f + '/' + str(month) + "/S1/" + str(band) + ".npy") for band in range(4) for month in range(12)])) for f in batch_filenames])
        #     batch_x_s2 = np.stack([np.array(np.concatenate([np.load(f + '/' + str(month) + "/S2/" + str(band) + ".npy") for band in range(11) for month in range(12)])) for f in batch_filenames])
        # except:
        #     print('missing data in there')
        # Return X, where X is made of loaded np arrays from the filenames
        # Shape s1 is 32 x 48 x 256 x 256 x 1 (all training arrays concatenated)
        # Shape s2 is 32 x 132 x 256 x 256 x 1 (all training arrays concatenated)
        # Directories divided in patches. Patches consist of 12 arrays for each month,
        # that for s1 contains 4 arrays and s2 11 arrays with format 256 x 256

        return (batch_x_s1, batch_x_s2), missing_s2, batch_y

class LocalDataGeneratorS(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=16):
        """
        Few things to mention:
            - The data generator tells our corrupted_model how to fetch one batch of training data (in this case from files)
            - Any work that can be done before training, should be done in init, since we want fetching a batch to be fast
            - Therefore, we want all filenames and labels to be determined before training
            - This saves work, because we will be fetching batches multiple times (across epochs)
        """
        # Get all filenames in directory
        self.filenames = [dir_path + file for file in os.listdir(dir_path)]
        print(self.filenames)
        print(len(self.filenames))
        # Include batch size as attribute
        self.batch_size = batch_size

    def __len__(self):
        """
        Should return the number of BATCHES the generator can retrieve (partial batch at end counts as well)
        """
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Tells generator how to retrieve BATCH idx
        """
        # Get filenames for X batch
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        label_blobs = []
        for file in batch_filenames:
            label = np.load(file + "/label.npy")
            if not label.any():
                batch_filenames.remove(file)
                continue
            label_blobs.append(label)

        batch_y = np.stack([f for f in label_blobs], axis=0)

        # concatenating all the months together and then stacking on each other for both s1 and s2 separate
        # List for all the patches together, to be processed to batch
        batch_s1_array = []
        batch_s2_array = []

        missing_s2 = []
        for file in batch_filenames:
            s1_data = []
            s2_data = []
            missing_s2_patch = []
            for month in range(12):
                missing = 0
                s1_temp = []
                for band in range(4):
                    s1_temp.append(np.load(file + '/' + str(month) + "/S1/" + str(band) + ".npy"))
                s1_data.append(s1_temp)

                s2_temp = []
                if path.exists(file + '/' + str(month) + "/S2/"):
                    for band in range(11):
                        s2_temp.append(np.load(file + '/' + str(month) + "/S2/" + str(band) + ".npy"))
                else:
                    missing = 1
                    for band in range(11):
                        fake_array_s2 = np.zeros((256, 256), dtype='uint16')
                        s2_temp.append(fake_array_s2)
                s2_data.append(s2_temp)
                missing_s2_patch.append(missing)

            batch_s1_array.append(np.concatenate(s1_data))
            batch_s2_array.append(np.concatenate(s2_data))

            missing_s2.append(missing_s2_patch)

        batch_x_s1 = np.stack([bs1 for bs1 in batch_s1_array])
        batch_x_s2 = np.stack([bs2 for bs2 in batch_s2_array])

        return (batch_x_s1, batch_x_s2), missing_s2, batch_y

if __name__ == "__main__":
    datagen = LocalDataGenerator(data_path)
    x, missing, y = datagen[1]
    print(x[0].shape, x[1].shape, missing)