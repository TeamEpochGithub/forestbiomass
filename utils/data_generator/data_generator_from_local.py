#Imports
import tensorflow as tf
from keras import datasets, layers, models

import numpy as np

import os

# from classification_models.tfkeras import Classifiers

np.random.seed(0)

data_path = "../../data/imgs"
train_features_path = "../../data/imgs/train_features"
train_abgm_path = "../../data/imgs/fake_data2/"

class LocalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=32):
        """
        Few things to mention:
            - The data generator tells our model how to fetch one batch of training data (in this case from files)
            - Any work that can be done before training, should be done in init, since we want fetching a batch to be fast
            - Therefore, we want all filenames and labels to be determined before training
            - This saves work, because we will be fetching batches multiple times (across epochs)
        """
        # Get all filenames in directory
        self.filenames = [dir_path + file for file in os.listdir(dir_path)]
        print(self.filenames)

        # Include batch size as attribute
        self.batch_size = batch_size

    def __len__(self):
        """
        Should return the number of BATCHES the generator can retrieve (partial batch at end counts as well)
        """
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    # def __getitem__(self, idx):
    #     """
    #     Tells generator how to retrieve BATCH idx
    #     """
    #     # Get filenames for X batch
    #     batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
    #
    #     batch_y = np.stack([np.load(f + "/label.npy") for f in batch_filenames], axis=0)
    #
    #     # concatenating all the months together and then stacking on each other
    #     batch_x = np.stack([np.array(np.concatenate([np.load(f + '/' + f.split('/')[-1] + "_" + str(month) + ".npy") for month in range(12)])) for f in batch_filenames])
    #     print(batch_x.shape)
    #
    #     # Return X, where X is made of loaded np arrays from the filenames
    #     # Shape is 32 x 180 x 256 x 256 x 1 (all training arrays concatenated)
    #     # Directories divided in patches. Patches consist of 12 arrays for each month,
    #     # that contain 15 arrays from both s1(4) and s2(11) with format 256 x 256
    #
    #     return batch_x, batch_y

    def __getitem__(self, idx):
        """
        Tells generator how to retrieve BATCH idx
        """
        # Get filenames for X batch
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = np.stack([np.load(f + "/label.npy") for f in batch_filenames], axis=0)

        # concatenating all the months together and then stacking on each other for both s1 and s2 separate
        batch_x_s1 = np.stack([np.array(np.concatenate([np.load(f + '/' + f.split('/')[-1] + "_S1_" + str(month) + ".npy") for month in range(12)])) for f in batch_filenames])
        batch_x_s2 = np.stack([np.array(np.concatenate([np.load(f + '/' + f.split('/')[-1] + "_S2_" + str(month) + ".npy") for month in range(12)])) for f in batch_filenames])
        print(batch_x_s1.shape)
        print(batch_x_s2.shape)
        # Return X, where X is made of loaded np arrays from the filenames
        # Shape s1 is 32 x 48 x 256 x 256 x 1 (all training arrays concatenated)
        # Shape s2 is 32 x 132 x 256 x 256 x 1 (all training arrays concatenated)
        # Directories divided in patches. Patches consist of 12 arrays for each month,
        # that for s1 contains 4 arrays and s2 11 arrays with format 256 x 256

        return (batch_x_s1, batch_x_s2), batch_y

if __name__ == "__main__":
    datagen = LocalDataGenerator(train_abgm_path)
    x, y = datagen[1]
    print(x)