#Imports
from google.api_core import page_iterator
import tensorflow as tf
from keras import datasets, layers, models
import google
from google.cloud import storage
import numpy as np
import sys
import os
import gslib
from tensorflow.python.lib.io import file_io
from io import StringIO, BytesIO

np.random.seed(0)

def authenticate_remote():
    """
    Method to authenticate to GSC
    :return: None or error
    """
    if sys.platform == "linux":
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../../keys/forestbiomass-key-epoch.json"
    elif sys.platform.startswith("win"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../../keys/forestbiomass-key-epoch.json"

    try:
        storage.Client()
        print("Authenticated successfully")
        return

    except google.auth.exceptions.DefaultCredentialsError as e:
        return e

def _item_to_value(iterator, item):
    return item
def list_directories(bucket_name, prefix):
    """
    Method to list subdirectories GCS, in this case all the patches
    :return: List of all the paths to the patches
    """
    if prefix and not prefix.endswith('/'):
        prefix += '/'

    extra_params = {
        "projection": "noAcl",
        "prefix": prefix,
        "delimiter": '/'
    }

    gcs = storage.Client()

    path = "/b/" + bucket_name + "/o"

    iterator = page_iterator.HTTPIterator(
        client=gcs,
        api_request=gcs._connection.api_request,
        path=path,
        items_key='prefixes',
        item_to_value=_item_to_value,
        extra_params=extra_params,
    )

    return [x for x in iterator]

class DataGenerator():
    def __init__(self, batch_size=2):
        """
        Few things to mention:
            - The data generator tells our model how to fetch one batch of training data (in this case from files)
            - Any work that can be done before training, should be done in init, since we want fetching a batch to be fast
            - Therefore, we want all filenames and labels to be determined before training
            - This saves work, because we will be fetching batches multiple times (across epochs)
        """
        # Get all filenames in directory
        self.filenames = list_directories('converted', 'forest')

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

        storage_client = storage.Client()
        bucket = storage_client.bucket("converted")
        print("Bucket ready")
        # Get filenames for X batch
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        label_blobs = []
        # For every filename in batch read the label.npy
        for file in batch_filenames:
            label = bucket.blob(f'{file}label.npy')
            label = BytesIO(label.download_as_string())
            l = np.load(label, allow_pickle=True)

            # Check if label.npy is corrupted or missing and remove if so
            if not l.any():
                batch_filenames.remove(file)
                continue
            label_blobs.append(l)

        batch_y = np.stack([label for label in label_blobs], axis=0)

        # concatenating all the months together and then stacking on each other for both s1 and s2 separate
        # List for all the patches together, to be processed to batch
        batch_s1_array = []
        batch_s2_array = []

        # Array for missing data. 0 means all data is present and 1 means missing data
        missing_data_s2 = []
        for file in batch_filenames:
            # List for all the data in one patch
            s1_blobs = []
            s2_blobs = []
            # Missing data per patch
            missing_data_patch_s2 = []
            for month in range(12):
                missing = 0
                # List all npy files for S1 and S2 separate per month
                s1_blobs_per_month = storage_client.list_blobs("converted", prefix=f'{file}{month}/S1')
                s2_blobs_per_month = storage_client.list_blobs("converted", prefix=f'{file}{month}/S2')
                # Proces data in blob to use np.load and append the whole month together in s1_blobs
                s1_temp = []
                for s1_blob in s1_blobs_per_month:
                    s1_blob = BytesIO(s1_blob.download_as_string())
                    s1_temp.append(np.load(s1_blob))
                s1_blobs.append(s1_temp)

                # Proces data in blob to use np.load and append the whole month together in s2_blobs
                s2_temp = []
                count = 0
                for s2_blob in s2_blobs_per_month:
                    count += 1
                    s2_blob = BytesIO(s2_blob.download_as_string())
                    s2_temp.append(np.load(s2_blob))

                if count == 0: # No blobs in S2
                    missing = 1
                    for band in range(11):
                        fake_array_s2 = np.zeros((256, 256), dtype='uint16')
                        s2_temp.append(fake_array_s2)
                s2_blobs.append(s2_temp)
                # Append information if data is missing or not
                missing_data_patch_s2.append(missing)

            # Concatenate all the months of one patch together and append it to the batch array
            batch_s2_array.append(np.concatenate(s2_blobs))
            batch_s1_array.append(np.concatenate(s1_blobs))

            # Append missing data per patch
            missing_data_s2.append(missing_data_patch_s2)

        # Stack all the patches on top of each other to create batch
        batch_x_s1 = np.stack([bs1 for bs1 in batch_s1_array])
        batch_x_s2 = np.stack([bs2 for bs2 in batch_s2_array])

        # Return X, where X is made of loaded np arrays from the filenames
        # Shape s1 is batch_size x 48 x 256 x 256 x 1 (all training arrays concatenated)
        # Shape s2 is batch_size x 132 x 256 x 256 x 1 (all training arrays concatenated)
        # Directories divided in patches. Patches consist of 12 arrays for each month,
        # that for s1 contains 4 arrays and s2 11 arrays with format 256 x 256
        print('End')
        return (batch_x_s1, batch_x_s2), missing_data_s2, batch_y

if __name__ == "__main__":
    authenticate_remote()

    path = "gs://converted/forest"

    datagen = DataGenerator()
    x, missing, y = datagen[8]
    print(x[0].shape, x[1].shape)