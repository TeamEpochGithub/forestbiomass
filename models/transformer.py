import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os.path as osp
import data
import csv
from sklearn.model_selection import train_test_split

from models.utils.get_train_data import get_average_green_band_data
from models.utils.root_mean_squared_error import root_mean_squared_error


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_model(input_shape, x_train, patch_size, num_patches, projection_dim, transformer_layers,
                     num_heads, transformer_units, mlp_head_units, learning_rate, weight_decay):
    inputs = layers.Input(shape=input_shape)
    # Augment data.

    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    data_augmentation.layers[0].adapt(x_train)

    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    unshaped = layers.Dense(256 * 256)(features)  # layers.Dense(num_classes)(features)
    logits = layers.Reshape((256, 256))(unshaped)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=root_mean_squared_error,
        metrics=[
            keras.metrics.MeanSquaredError(name="MSE"),
            keras.metrics.RootMeanSquaredError(name="RMSE"),
        ],
    )

    return model


def fit_vit_model(model, x_train, y_train, batch_size, num_epochs, validation_split, save_checkpoint=False):
    if save_checkpoint:
        checkpoint_filepath = "./vit_checkpoints/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="RMSE",
            save_best_only=True,
            save_weights_only=True,
        )
    else:
        checkpoint_callback = ()

    model.fit(
        x=x_train,
        y=y_train.clip(min=0, max=300),
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=validation_split,
        callbacks=[checkpoint_callback],
    )

    return model


def get_trained_model(input_shape, x_train, patch_size, num_patches, projection_dim, transformer_layers,
                      num_heads, transformer_units, mlp_head_units, learning_rate, weight_decay):
    model = create_vit_model(input_shape, x_train, patch_size, num_patches, projection_dim,
                             transformer_layers,
                             num_heads, transformer_units, mlp_head_units, learning_rate, weight_decay)
    model.load_weights("./vit_checkpoints/checkpoint")
    return model


def evaluate_vit_model(model, x_test, y_test, use_checkpoint=False):
    if use_checkpoint:
        model.load_weights("./vit_checkpoints/checkpoint")

    _, mse, rmse = model.evaluate(x_test, y_test)
    return mse, rmse


if __name__ == '__main__':
    learning_rate = 0.001
    weight_decay = 0.0001
    input_shape = (256, 256, 1)
    batch_size = 16  # 256
    num_epochs = 3
    image_size = 256  # 72  # We'll resize input images to this size
    patch_size = 16  # 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 16
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 16
    mlp_head_units = [4096, 2048, 1024]  # [2048, 1024]  # Size of the dense layers of the final classifier

    print("Getting train data...")
    train_data_path = osp.join(osp.dirname(data.__file__), "converted")
    with open(osp.join(osp.dirname(data.__file__), 'patch_names'), newline='') as f:
        reader = csv.reader(f)
        patch_name_data = list(reader)
    patch_names = patch_name_data[0]

    X_all, y_all = get_average_green_band_data(patch_names, train_data_path)

    x_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    x_train = x_train.reshape(x_train.shape[0], 256, 256, 1)
    x_test = x_test.reshape(x_test.shape[0], 256, 256, 1)

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

    model = create_vit_model(input_shape=input_shape,
                             x_train=x_train,
                             patch_size=patch_size,
                             num_patches=num_patches,
                             projection_dim=projection_dim,
                             transformer_layers=transformer_layers,
                             num_heads=num_heads,
                             transformer_units=transformer_units,
                             mlp_head_units=mlp_head_units,
                             learning_rate=learning_rate,
                             weight_decay=weight_decay)

    fitted_model = fit_vit_model(model=model, x_train=x_train, y_train=y_train, batch_size=batch_size,
                                 num_epochs=num_epochs, validation_split=0.1, save_checkpoint=False)
    mse, rmse = evaluate_vit_model(model=fitted_model, x_test=x_test, y_test=y_test, use_checkpoint=False)
    print(f"MSE: {round(mse, 2)} and RMSE: {round(rmse, 2)}")

    pretrained_model = get_trained_model(input_shape=input_shape,
                                         x_train=x_train,
                                         patch_size=patch_size,
                                         num_patches=num_patches,
                                         projection_dim=projection_dim,
                                         transformer_layers=transformer_layers,
                                         num_heads=num_heads,
                                         transformer_units=transformer_units,
                                         mlp_head_units=mlp_head_units,
                                         learning_rate=learning_rate,
                                         weight_decay=weight_decay)

    mse, rmse = evaluate_vit_model(model=pretrained_model, x_test=x_test, y_test=y_test, use_checkpoint=False)
    print(f"MSE: {round(mse, 2)} and RMSE: {round(rmse, 2)}")
