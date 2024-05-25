import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(final_data):
    X_train, X_test, Y_train, Y_test = train_test_split(
        final_data, final_data['disease'], stratify=final_data['disease'], random_state=43, test_size=0.25)
    return X_train, X_test, Y_train, Y_test

def encode_labels(Y_train, Y_test):
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_test = le.transform(Y_test)
    return Y_train, Y_test, le

def train_model(model, mfcc_train, cstft_train, mSpec_train, Y_train, mfcc_test, cstft_test, mSpec_test, Y_test):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        {'mfcc': mfcc_train, 'cstft': cstft_train, 'mspec': mSpec_train},
        Y_train,
        validation_data=({'mfcc': mfcc_test, 'cstft': cstft_test, 'mspec': mSpec_test}, Y_test),
        epochs=100,
        callbacks=callbacks,
        batch_size=32
    )
    return history

def plot_history(history):
    sns.set()
    plt.figure(figsize=(20, 4))
    plt.plot(history.history['loss'], color='blue', label='Training loss')
    plt.plot(history.history['val_loss'], color='orange', label='Validation loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 4))
    plt.plot(history.history['accuracy'], color='blue', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='Validation accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.show()

# Example usage:
# X_train, X_test, Y_train, Y_test = prepare_data(final_data)
# Y_train, Y_test, le = encode_labels(Y_train, Y_test)
# history = train_model(model, mfcc_train, cstft_train, mSpec_train, Y_train, mfcc_test, cstft_test, mSpec_test, Y_test)
# plot_history(history)
