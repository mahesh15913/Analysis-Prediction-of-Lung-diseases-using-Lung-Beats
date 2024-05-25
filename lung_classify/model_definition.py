from tensorflow import keras

def build_cnn_model(input_shape, name):
    input_layer = keras.layers.Input(shape=input_shape, name=f"{name}Input")
    x = keras.layers.Conv2D(32, 5, strides=(1, 3), padding='same')(input_layer)
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
    output_layer = keras.layers.GlobalMaxPooling2D()(x)

    return keras.Model(input_layer, output_layer, name=f"{name}Model")

def build_combined_model(input_shapes):
    mfcc_model = build_cnn_model(input_shapes['mfcc'], 'mfcc')
    cstft_model = build_cnn_model(input_shapes['cstft'], 'cstft')
    mspec_model = build_cnn_model(input_shapes['mspec'], 'mspec')

    input_mfcc = keras.layers.Input(shape=input_shapes['mfcc'], name="mfcc")
    input_cstft = keras.layers.Input(shape=input_shapes['cstft'], name="cstft")
    input_mSpec = keras.layers.Input(shape=input_shapes['mspec'], name="mspec")

    mfcc_output = mfcc_model(input_mfcc)
    cstft_output = cstft_model(input_cstft)
    mspec_output = mspec_model(input_mSpec)

    concat = keras.layers.concatenate([mfcc_output, cstft_output, mspec_output])
    hidden = keras.layers.Dropout(0.2)(concat)
    hidden = keras.layers.Dense(50, activation='relu')(hidden)
    hidden = keras.layers.Dropout(0.3)(hidden)
    hidden = keras.layers.Dense(25, activation='relu')(hidden)
    hidden = keras.layers.Dropout(0.3)(hidden)
    output = keras.layers.Dense(8, activation='softmax')(hidden)

    model = keras.Model([input_mfcc, input_cstft, input_mSpec], output, name="Net")
    return model

# Example usage:
# input_shapes = {'mfcc': (20, 259, 1), 'cstft': (12, 259, 1), 'mspec': (128, 259, 1)}
# model = build_combined_model(input_shapes)
# model.summary()
