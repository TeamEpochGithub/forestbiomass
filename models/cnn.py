from osgeo import gdal
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import os.path as osp
import csv
import data


class CNNModel(tf.keras.Model):
    def __init__(self, base, weights):
        super(CNNModel, self).__init__()
        self.base = base
        base_weights = weights

        for i in range(3, len(self.base.layers)):
            self.base.layers[i].set_weights(base_weights.layers[i].get_weights())

        del base_weights
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dens3 = tf.keras.layers.Dense(256, activation='relu')
        self.out = tf.keras.layers.Dense(256 * 256, activation='linear')

    def call(self, x, training=None, **kwargs):
        x = self.base(x)
        x = self.pool(x)
        x = self.dens3(x)
        x = self.out(x)
        return x


def create_model(base_model, base_weights):
    model = CNNModel(base_model, base_weights)
    model.build(input_shape=(None, 256, 256, 11))
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def get_patch_names():
    with open(osp.join(osp.dirname(data.__file__), 'patch_name_test'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)

    return patch_name_data[0]

class DataGeneratorTif(tf.keras.utils.Sequence):
    def __init__(self, patch_ids, batch_size=4):
        """
        Few things to mention:
            - This data generator uses original tif files, now only s2 and averages them for every patch
            - Data is reshaped to be compatible with CNN
            - The data generator tells our model how to fetch one batch of training data (in this case from files)
            - Any work that can be done before training, should be done in init, since we want fetching a batch to be fast
            - Therefore, we want all filenames and labels to be determined before training
            - This saves work, because we will be fetching batches multiple times (across epochs)
        """
        self.train_dir_path = r'C:\Users\kuipe\OneDrive\Bureaublad\Epoch\forestbiomass\data\train_features/'
        self.label_dir = r'C:\Users\kuipe\OneDrive\Bureaublad\Epoch\forestbiomass\data\train_agbm/'
        # Get all filenames in directory
        self.patches = patch_ids

        # Include batch size as attribute
        self.batch_size = batch_size

    def __len__(self):
        """
        Should return the number of BATCHES the generator can retrieve (partial batch at end counts as well)
        """
        return int(np.ceil(len(self.patches) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Tells generator how to retrieve BATCH idx
        """

        # Get filenames for X batch
        batch_patches = self.patches[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = [self.label_dir + p + '_agbm.tif' for p in batch_patches]
        # batch_filenames = [self.train_dir_path + p + f'_S2_{month}.tif' for p in batch_patches for month in range(12)]
        batch_filenames = [osp.join(self.train_dir_path, file) for file in os.listdir(self.train_dir_path) for p in batch_patches if f'{p}_S2' in file]
        # print(batch_filenames)

        batch_x = []
        batch_y = []
        for label_tif in batch_labels:
            label = gdal.Open(label_tif)
            label = label.ReadAsArray()
            label = label.reshape(256 * 256)
            batch_y.append(label)
        # For every patch in batch get all possible s2 files
        for p in batch_patches:
            batch_per_patch = []
            for s2_tif in batch_filenames:
                if p in s2_tif:
                    s2 = gdal.Open(s2_tif)
                    s2 = s2.ReadAsArray()
                    s2 = s2.reshape(256, 256, 11)
                    batch_per_patch.append(s2)
            # Average s2 months to one array
            batch_x.append(np.average(batch_per_patch, axis=0))

        return np.asarray(batch_x), np.asarray(batch_y)

class DataGeneratorNpy(tf.keras.utils.Sequence):
    def __init__(self, patch_ids, batch_size=4):
        """
        Few things to mention:
            - The data generator tells our model how to fetch one batch of training data (in this case from files)
            - Any work that can be done before training, should be done in init, since we want fetching a batch to be fast
            - Therefore, we want all filenames and labels to be determined before training
            - This saves work, because we will be fetching batches multiple times (across epochs)
        """
        self.dir_path = r'C:\Users\kuipe\OneDrive\Bureaublad\Epoch\forestbiomass\data\test/'
        # Get all filenames in directory
        self.patches = patch_ids
        # Include batch size as attribute
        self.batch_size = batch_size

    def __len__(self):
        """
        Should return the number of BATCHES the generator can retrieve (partial batch at end counts as well)
        """
        return int(np.ceil(len(self.patches) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Tells generator how to retrieve BATCH idx
        """

        # Get filenames for X batch
        batch_patches = self.patches[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = [self.dir_path + p + '/label.npy' for p in batch_patches]

        batch_x = []
        batch_y = []
        # Get labels per patch.
        for label_npy in batch_labels:
            label = np.load(label_npy).reshape(256 * 256)
            batch_y.append(label)

        # Get all possible s2 bands that are available per patch
        for p in batch_patches:

            s2_patch = []
            for month in range(12):
                s2_month = []

                # Check if S2 data exists for a specific month
                try:
                    bands = [osp.join(self.dir_path + p + f'/{month}/S2/', file) for file in os.listdir(self.dir_path + p + f'/{month}/S2')]
                    for band in bands:
                        s2_month.append(np.load(band))
                except:
                    continue

                s2_patch.append(s2_month)

            # average all the bands per patch together.
            average = np.average(s2_patch, axis=0)
            # Reshape it so that the CNN model can take the data in. 11 is the number of channels.
            average = average.reshape(256, 256, 11)
            batch_x.append(average)

        return np.asarray(batch_x), np.asarray(batch_y)

def create_data_generator(mode):


if __name__ == '__main__':
    # Change this to some pretrained cnn of keras
    # Check this link for different models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_shape=(256, 256, 11))
    base_weights = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    # Create model
    model = create_model(base_model, base_weights)
    # Get patch names to feed in data generator.
    patch_names = get_patch_names()
    datagen = DataGeneratorNpy(patch_names)
    # Use data generator to fit on model
    model.fit(datagen, epochs=100)