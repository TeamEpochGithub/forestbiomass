#Imports
import tensorflow as tf
from keras import datasets, layers, models
import google
from google.cloud import storage
import numpy as np
import sys
import os

# from classification_models.tfkeras import Classifiers

def authenticate_remote():
    """
    Method to authenticate to GSC
    :return: None or error
    """
    if sys.platform == "linux":
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../../keys/forestbiomass-key-sietse.json"
    elif sys.platform.startswith("win"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../../keys/forestbiomass-key-sietse.json"

    try:
        storage.Client()
        return

    except google.auth.exceptions.DefaultCredentialsError as e:
        return e

np.random.seed(0)



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir_path, batch_size=3):
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
        try:
            batch_y = np.stack([np.load(f + "/label.npy") for f in batch_filenames], axis=0)
        except:
            print(f"something wrong with label")

        # concatenating all the months together and then stacking on each other for both s1 and s2 separate
        try:
            batch_x_s1 = np.stack([np.array(np.concatenate([np.load(f + '/' + str(month) + "/S1/" + str(band) + ".npy") for band in range(4) for month in range(12)])) for f in batch_filenames])
            batch_x_s2 = np.stack([np.array(np.concatenate([np.load(f + '/' + str(month) + "/S2/" + str(band) + ".npy") for band in range(11) for month in range(12)])) for f in batch_filenames])
        except:
            print('missing data in there')
        # Return X, where X is made of loaded np arrays from the filenames
        # Shape s1 is 32 x 48 x 256 x 256 x 1 (all training arrays concatenated)
        # Shape s2 is 32 x 132 x 256 x 256 x 1 (all training arrays concatenated)
        # Directories divided in patches. Patches consist of 12 arrays for each month,
        # that for s1 contains 4 arrays and s2 11 arrays with format 256 x 256

        return (batch_x_s1, batch_x_s2), batch_y


if __name__ == "__main__":
    authenticate_remote()

    storage_client = storage.Client()
    bucket = storage_client.bucket("biomass-data")

    # blobs = storage_client.list_blobs("biomass-data")
    #
    # # Note: The call returns a response only when the iterator is consumed.
    # for blob in blobs:
    #     print(blob.name)


    bucket_path = "gs://biomass-data/"

    datagen = DataGenerator(bucket_path)
    x, y = datagen[0]
    # print(x.shape)