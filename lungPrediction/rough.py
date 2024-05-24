import pandas as pd
import numpy as np
import os
import tqdm
from os import listdir
import librosa
import soundfile as sf
import shutil


directory = "./Analysis-Prediction-of-Lung-diseases-using-Lung-Beats/Respiratory sounds"


content = listdir(directory)

content[0:10] #Printed some data stuff just a check

patient_lung_data = []
l = [s for s in content if '.txt' in s]
for i in l:
  for lines in open(directory+'/'+i,'r').readlines():
    patient_lung_data.append([i.split('.')[0].split("_")[0],i.split('.')[0],lines.split()[0],lines.split()[1],lines.split()[2],lines.split()[3]])


k = 0
for m in patient_lung_data:
  if m[0] == '151':
    k += 1


def patient(x):
  patient_list = []
  patient_id = x.split('.')[0].split("_")[0]
  recording_index = x.split('.')[0].split("_")[1]
  chest_location = x.split('.')[0].split("_")[2]
  acquisition_mode = x.split('.')[0].split("_")[3]
  recording_equipment = x.split('.')[0].split("_")[4]
  filename = x.split('.')[0]

  patient_list.append(patient_id)
  patient_list.append(recording_index)
  patient_list.append(chest_location)
  patient_list.append(acquisition_mode)
  patient_list.append(recording_equipment)
  patient_list.append(filename)

  return patient_list


df = [patient(s) for s in content if '.txt' in s]



data = pd.DataFrame(df)


data.head()

data.columns = ['patient_id','recording_index','chest_location','acquisition_mode','recording_equipment' ,'filename']

data.head()


data['chest_location'].unique()


data['recording_equipment'].unique()


data['recording_index'].unique()


print(data[data['patient_id']== '151'])


data.info()


data[:10]


lung_data = pd.DataFrame(patient_lung_data)
lung_data.head()


lung_data.columns = ['patient_id','filename','start','end','crackles','wheezes']

lung_data.head(10)

datax = pd.concat([data,lung_data], join = 'inner', axis = 1)

datax


datay = data.merge(lung_data, how = 'outer',on = 'patient_id' and  'filename')
datay


datay[datay['patient_id_x'] == '160'].head(15)


datay.drop_duplicates()


patient_demographic = pd.read_csv('C:/Users/mahii/OneDrive/Desktop/practice/MajorProject/Respiratory sounds/patient_diagnosis.csv')
patient_demographic.head(10)


patient_demographic.columns = ['patient_id','disease']



patient_demographic.head()



patient_demographic.info()


c1 = pd.DataFrame({'patient_id':[101],
                   'disease':['URTI']})
c1.info()



patient_demographic = pd.concat([patient_demographic,c1])



patient_demographic['patient_id'].unique()



datay = datay.drop('patient_id_y',axis= 1)


datay.rename(columns = {'patient_id_x':'patient_id'}, inplace = True)


datay.head()

daty = datay.convert_dtypes()


datay.info()

datay['patient_id'] = datay['patient_id'].astype('int64')


datay['patient_id'].dtype

patient_demographic[patient_demographic['patient_id'] == 101]


datay['start'] = datay['start'].astype('float64')
datay['end'] = datay['end'].astype('float64')
datay['crackles'] = datay['crackles'].astype('int64')
datay['wheezes'] = datay['wheezes'].astype('int64')


datay.info()



def recording_to_clips(root, start, end, sr = 44100):

  maximum_index = len(root)
  start_index = min(int(start*sr), maximum_index)
  end_index = min(int(end*sr), maximum_index)
  return root[start_index:end_index]



os.makedirs('/content/sample_data/processed_clips')



i = 0
n = 0
for index, rows in tqdm.tqdm(datay.iterrows()):
  max_length = 6
  start_time = rows['start']
  end_time = rows['end']
  filename = rows['filename']

  if end_time - start_time > max_length:
    end = start_time + max_length

  audio_file_path = directory + '/' + filename + '.wav'

  if index > 0:
    if datay.iloc[index-1]['filename'] == filename:
      i += 1
    else:
      i = 0

  filename = filename + '_' + str(i) + '.wav'
  save_path = '/content/sample_data/processed_clips/'+filename
  n += 1

  audio, sample_rate = librosa.load(audio_file_path)
  sample = recording_to_clips(audio,start_time,end_time,sample_rate)

  required_length = max( 6*sample_rate, len(sample))
  padded_data = librosa.util.pad_center(sample,size = required_length)

  sf.write(file = save_path,data = padded_data, samplerate = sample_rate)

  # print('total files processed:', n)


path = '/content/sample_data/processed_clips/'


lll = [[s.split("_")[0],s] for s in os.listdir(path = path)]
patient_data = pd.DataFrame(lll,columns = ['patient_id','filename'])
patient_data


patient_data['patient_id'] = patient_data['patient_id'].astype('int64')

patient_data.info()

final_data = patient_data.merge(patient_demographic, how = 'outer', on = 'patient_id')
final_data

final_data[['patient_id','disease']].value_counts()

import sklearn as sk
from sklearn.model_selection import train_test_split

X_train,X_test, Y_train, Y_test = train_test_split(final_data, final_data['disease'],stratify  = final_data['disease'], random_state = 43, test_size = 0.25)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

X_train.head()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Y_train = le.fit_transform(Y_train)
Y_test = le.fit_transform(Y_test)


def feature_shape(feature):
  ushape = [np.shape(sub_list) for sub_list in feature]
  return set(ushape)


def feature_padding(feature):
  min_shape = min(sub_array.shape for sub_list in feature for sub_array in sub_list)
  print(min_shape, type(min_shape))
  # Padding each subarray to the minimum shape with zeros
  feature_padded = [[sub_array[:min_shape[0]] for sub_array in sub_list] for sub_list in feature]
  return feature_padded

def get_features_from_audio(path):
    soundArr,sample_rate= librosa.load(path)
    mfcc=librosa.feature.mfcc(y=soundArr,sr=sample_rate)
    cstft=librosa.feature.chroma_stft(y=soundArr,sr=sample_rate)
    mSpec=librosa.feature.melspectrogram(y=soundArr,sr=sample_rate)
    tone = librosa.feature.tonnetz(y=soundArr,sr=sample_rate)
    specCen = librosa.feature.spectral_centroid(y=soundArr,sr=sample_rate)

    return mfcc,cstft,mSpec,tone, specCen



root = '/content/sample_data/processed_clips/'
mfcc,cstft,mSpec=[],[],[]

for idx,row in tqdm.tqdm(X_test.iterrows()):
    path=root + row['filename']
    a,b,c,d,e=get_features_from_audio(path)
    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)



print(feature_shape(mfcc))


print(feature_shape(cstft))


print(feature_shape(mSpec))

mfcc_padded = feature_padding(mfcc)

cstft_padded = feature_padding(cstft)


mSpec_padded = feature_padding(mSpec)



mfcc_test = np.array(mfcc_padded)
cstft_test = np.array(cstft_padded)
mSpec_test = np.array(mSpec_padded)

print(f"mfcc: {mfcc_test.shape}")
print(f"cstft: {cstft_test.shape}")
print(f"mSpec: {mSpec_test.shape}")


root = '/content/sample_data/processed_clips/'
mfcc,cstft,mSpec=[],[],[]

for idx,row in tqdm.tqdm(X_train.iterrows()):
    path=root + row['filename']
    a,b,c,d,e=get_features_from_audio(path)
    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)


mfcc_shape = feature_shape(mfcc)
print(mfcc_shape)


cstft_shape = feature_shape(cstft)
print(cstft_shape)

mspec = feature_shape(mSpec)
print(mspec)


mfcc_padded = feature_padding(mfcc)
cstft_padded = feature_padding(cstft)
mSpec_padded = feature_padding(mSpec)


mfcc_train = np.array(mfcc_padded)
cstft_train = np.array(cstft_padded)
mSpec_train = np.array(mSpec_padded)

print(mfcc_train.shape)
print(cstft_train.shape)
print(mSpec_train.shape)


import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
),
    tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5,
    verbose=1,
    mode="min",
    min_lr=0.00001,
)
]



sns.countplot(x = final_data['disease'],palette = "pastel")
plt.xticks(rotation = 45)


sns.countplot(x = patient_demographic['disease'],palette = "pastel")
plt.xticks(rotation = 45)


mfcc_input=keras.layers.Input(shape=(20,259,1),name="mfccInput")
x=keras.layers.Conv2D(32,5,strides=(1,3),padding='same')(mfcc_input)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(64,3,strides=(1,2),padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(96,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(128,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
mfcc_output=keras.layers.GlobalMaxPooling2D()(x)

mfcc_model=keras.Model(mfcc_input, mfcc_output, name="mfccModel")



mfcc_model.summary()


cstft_input=keras.layers.Input(shape=(12,259,1),name="cstftInput")
x=keras.layers.Conv2D(32,5,strides=(1,3),padding='same')(cstft_input)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(64,3,strides=(1,2),padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(96,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(128,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
cstft_output=keras.layers.GlobalMaxPooling2D()(x)

cstft_model=keras.Model(cstft_input, cstft_output, name="cstftModel")


cstft_model.summary()


mspec_input=keras.layers.Input(shape=(128,259,1),name="mspecInput")
x=keras.layers.Conv2D(32,5,strides=(1,3),padding='same')(mspec_input)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(64,3,strides=(1,2),padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(96,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)

x=keras.layers.Conv2D(128,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
mspec_output=keras.layers.GlobalMaxPooling2D()(x)

mspec_model=keras.Model(mspec_input, mspec_output, name="mspecModel")

print(type(mspec_model))

mspec_model.summary()

input_mfcc=keras.layers.Input(shape=(20,259,1),name="mfcc")
mfcc=mfcc_model(input_mfcc)

input_cstft=keras.layers.Input(shape=(12,259,1),name="cstft")
cstft=cstft_model(input_cstft)

input_mSpec=keras.layers.Input(shape=(128,259,1),name="mspec")
mSpec=mspec_model(input_mSpec)


concat=keras.layers.concatenate([mfcc,cstft,mSpec])
hidden=keras.layers.Dropout(0.2)(concat)
hidden=keras.layers.Dense(50,activation='relu')(concat)
hidden=keras.layers.Dropout(0.3)(hidden)
hidden=keras.layers.Dense(25,activation='relu')(hidden)
hidden=keras.layers.Dropout(0.3)(hidden)
output=keras.layers.Dense(8,activation='softmax')(hidden)

net=keras.Model([input_mfcc,input_cstft,input_mSpec], output, name="Net")

print(type(mfcc))

net.summary()

import keras
import tensorflow as tf

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)

net.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# tf.keras.backend.set_value(net.optimizer.learning_rate, 0.001)


history=net.fit(
    {"mfcc":mfcc_train,"cstft":cstft_train,"mspec":mSpec_train},
    Y_train,
    validation_data=({"mfcc":mfcc_test,"cstft":cstft_test,"mspec":mSpec_test},Y_test),
    epochs=100,verbose=1,
    callbacks=my_callbacks
)



for i in mfcc_test:
    y_pred = net.predict(i)
    print(y_pred)



from sklearn.metrics import accuracy_score, confusion_matrix

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()







# =keras.layers.Input(shape=(20,259,1),name="mfcc")
mfcc=mfcc_model(input_mfcc)



hidden=keras.layers.Dropout(0.2)(mfcc)
hidden=keras.layers.Dense(50,activation='relu')(mfcc)
hidden=keras.layers.Dropout(0.3)(hidden)
hidden=keras.layers.Dense(25,activation='relu')(hidden)
hidden=keras.layers.Dropout(0.3)(hidden)
output=keras.layers.Dense(8,activation='softmax')(hidden)

net1=keras.Model(input_mfcc, output, name="Net")


optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)

net1.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



history1=net1.fit(
    mfcc_train,
    Y_train,
    validation_data=(mfcc_test,Y_test),
    epochs=100,verbose=1,
    callbacks=my_callbacks
)


pd.DataFrame(history1.history).plot()
plt.grid(True)
plt.show()

y_pred1 = net1.predict(mfcc_test)


print(accuracy_score(y_pred1,Y_test))


shutil.rmtree('/content/sample_data/processed_clips')
