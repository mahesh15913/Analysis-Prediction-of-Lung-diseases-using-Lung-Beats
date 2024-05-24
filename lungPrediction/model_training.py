import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def build_model():
    mfcc_input = keras.layers.Input(shape=(20, 259, 1), name="mfccInput")
    x = keras.layers.Conv2D(32, 5, strides=(1, 3), padding='same')(mfcc_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(64, 3, strides=(1, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(96, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(128, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    mfcc_output = keras.layers.GlobalMaxPooling2D()(x)

    mfcc_model = keras.Model(mfcc_input, mfcc_output, name="mfccModel")

    cstft_input = keras.layers.Input(shape=(12, 259, 1), name="cstftInput")
    x = keras.layers.Conv2D(32, 5, strides=(1, 3), padding='same')(cstft_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(64, 3, strides=(1, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(96, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(128, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    cstft_output = keras.layers.GlobalMaxPooling2D()(x)

    cstft_model = keras.Model(cstft_input, cstft_output, name="cstftModel")

    mspec_input = keras.layers.Input(shape=(128, 259, 1), name="mspecInput")
    x = keras.layers.Conv2D(32, 5, strides=(1, 3), padding='same')(mspec_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(64, 3, strides=(1, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(96, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding='valid')(x)

    x = keras.layers.Conv2D(128, 2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    mspec_output = keras.layers.GlobalMaxPooling2D()(x)

    mspec_model = keras.Model(mspec_input, mspec_output, name="mspecModel")

    combined = keras.layers.concatenate([mfcc_model.output, cstft_model.output, mspec_model.output])
    z = keras.layers.Dense(256, activation="relu")(combined)
    z = keras.layers.Dropout(0.3)(z)
    z = keras.layers.Dense(128, activation="relu")(z)
    z = keras.layers.Dropout(0.3)(z)
    z = keras.layers.Dense(32, activation="relu")(z)
    z = keras.layers.Dense(8, activation="softmax")(z)

    model = keras.Model(inputs=[mfcc_model.input, cstft_model.input, mspec_model.input], outputs=z, name="FinalModel")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

def train_model(model, mfcc_train, cstft_train, mSpec_train, Y_train, mfcc_test, cstft_test, mSpec_test, Y_test):
    history = model.fit(
        x={"mfccInput": mfcc_train, "cstftInput": cstft_train, "mspecInput": mSpec_train},
        y=Y_train,
        epochs=30,
        batch_size=32,
        validation_data=(
            {"mfccInput": mfcc_test, "cstftInput": cstft_test, "mspecInput": mSpec_test},
            Y_test
        )
    )
    return history

def evaluate_model(model, mfcc_test, cstft_test, mSpec_test, Y_test):
    return model.evaluate(
        x={"mfccInput": mfcc_test, "cstftInput": cstft_test, "mspecInput": mSpec_test},
        y=Y_test
    )

def plot_history(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
