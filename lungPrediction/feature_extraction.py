import numpy as np
import librosa
import soundfile as sf
import os
import tqdm

def recording_to_clips(root, start, end, sr=44100):
    maximum_index = len(root)
    start_index = min(int(start*sr), maximum_index)
    end_index = min(int(end*sr), maximum_index)
    return root[start_index:end_index]

def extract_clips(datay, directory, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for index, rows in tqdm.tqdm(datay.iterrows()):
        max_length = 6
        start_time = rows['start']
        end_time = rows['end']
        filename = rows['filename']

        if end_time - start_time > max_length:
            end = start_time + max_length

        audio_file_path = directory + '/' + filename + '.wav'

        if index > 0:
            if datay.iloc[index - 1]['filename'] == filename:
                i += 1
            else:
                i = 0

        filename = filename + '_' + str(i) + '.wav'
        save_path = save_dir + filename

        audio, sample_rate = librosa.load(audio_file_path)
        sample = recording_to_clips(audio, start_time, end_time, sample_rate)

        required_length = max(6*sample_rate, len(sample))
        padded_data = librosa.util.pad_center(sample, size=required_length)

        sf.write(file=save_path, data=padded_data, samplerate=sample_rate)

def get_features_from_audio(path):
    soundArr, sample_rate = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=soundArr, sr=sample_rate)
    cstft = librosa.feature.chroma_stft(y=soundArr, sr=sample_rate)
    mSpec = librosa.feature.melspectrogram(y=soundArr, sr=sample_rate)
    tone = librosa.feature.tonnetz(y=soundArr, sr=sample_rate)
    specCen = librosa.feature.spectral_centroid(y=soundArr, sr=sample_rate)
    return mfcc, cstft, mSpec, tone, specCen

def extract_features(X, root):
    mfcc, cstft, mSpec = [], [], []
    for idx, row in tqdm.tqdm(X.iterrows()):
        path = root + row['filename']
        a, b, c, d, e = get_features_from_audio(path)
        mfcc.append(a)
        cstft.append(b)
        mSpec.append(c)
    return mfcc, cstft, mSpec

def feature_padding(feature):
    min_shape = min(sub_array.shape for sub_list in feature for sub_array in sub_list)
    feature_padded = [[sub_array[:min_shape[0]] for sub_array in sub_list] for sub_list in feature]
    return feature_padded
