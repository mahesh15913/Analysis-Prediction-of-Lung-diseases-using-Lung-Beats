import librosa
import numpy as np
import pandas as pd
import os

def get_features_from_audio(path):
    sound_arr, sample_rate = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=sound_arr, sr=sample_rate)
    cstft = librosa.feature.chroma_stft(y=sound_arr, sr=sample_rate)
    mSpec = librosa.feature.melspectrogram(y=sound_arr, sr=sample_rate)
    tone = librosa.feature.tonnetz(y=sound_arr, sr=sample_rate)
    specCen = librosa.feature.spectral_centroid(y=sound_arr, sr=sample_rate)
    return mfcc, cstft, mSpec, tone, specCen

def extract_features(data, root):
    mfcc, cstft, mSpec = [], [], []
    for idx, row in data.iterrows():
        path = os.path.join(root, row['filename'])
        a, b, c, d, e = get_features_from_audio(path)
        mfcc.append(a)
        cstft.append(b)
        mSpec.append(c)
    return np.array(mfcc), np.array(cstft), np.array(mSpec)

def feature_padding(feature):
    min_shape = min(sub_array.shape for sub_list in feature for sub_array in sub_list)
    feature_padded = [[sub_array[:min_shape[0]] for sub_array in sub_list] for sub_list in feature]
    return np.array(feature_padded)

# Example usage:
# root = "path_to_processed_clips"
# mfcc, cstft, mSpec = extract_features(data, root)
# mfcc_padded = feature_padding(mfcc)
# cstft_padded = feature_padding(cstft)
# mSpec_padded = feature_padding(mSpec)
