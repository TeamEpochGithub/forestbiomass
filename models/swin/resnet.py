import keras.utils
import tensorflow as tf
from keras import datasets, layers, models
import numpy as np
import pandas as pd
import os

from classification_models.tfkeras import Classifiers


def resnet(deep_layers: int,
           input_shape: tuple,
           weights: str = 'imagenet'):
    """
    We want to use a pretrained resnet34 (many other models available from the package we use)
    For our task, we take off the layers responsible for performing the task, but keep the deep convolutional layers
    These are trained on extracting features, which is very useful for our data
    :param deep_layers: Number of deep layers (at least supported: 18, 34, 50, 101, 151)
    :param input_shape: Shape of input data without batch size (should be width, height, 3)
    :param weights: Where to get weights from. Default pretrained imagenets
    :return: Model
    """

    # Load the pretrained model
    loaded_resnet, preprocess_input = Classifiers.get(f'resnet{deep_layers}')

    # Get the Resnet model, provide input shape of our data and specify on what problem it should be pretrained
    base_model = loaded_resnet(input_shape=input_shape, weights=weights, include_top=False)

    # Construct our output layers at the end of the pretrained model
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    # And specify multiplication with the previous layer instantly
    flatten = tf.keras.layers.Dense(256, activation='relu')(x)

    # Construct the actual output layer
    # We will use a linear layer (which basically does nothing)
    output = tf.keras.layers.Dense(256 * 256, activation='linear')(flatten)

    # Construct the model from the pretrained (34) deep layers, and the specified output layers
    model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])

    # Compile the model
    # We will use Stochastic Gradient Descent and MAE as loss, since this was the competition  evaluation metric
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    # You could print the model summary, but it will be very long ;)
    model.summary()

    return model


if __name__ == '__main__':
    resnet(18, (256, 256, 3))

