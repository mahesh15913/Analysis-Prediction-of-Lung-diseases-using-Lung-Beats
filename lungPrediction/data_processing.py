import pandas as pd
import os

def load_data(directory):
    content = os.listdir(directory)
    return content

def parse_patient_lung_data(content, directory):
    patient_lung_data = []
    txt_files = [s for s in content if '.txt' in s]
    for i in txt_files:
        for lines in open(directory + '/' + i, 'r').readlines():
            patient_lung_data.append([
                i.split('.')[0].split("_")[0], i.split('.')[0],
                lines.split()[0], lines.split()[1], lines.split()[2], lines.split()[3]
            ])
    return patient_lung_data

def create_patient_dataframe(content):
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
    data.columns = ['patient_id', 'recording_index', 'chest_location', 'acquisition_mode', 'recording_equipment', 'filename']
    return data

def merge_dataframes(data, lung_data):
    return pd.concat([data, lung_data], join='inner', axis=1)

def load_patient_demographic(file_path):
    patient_demographic = pd.read_csv(file_path)
    patient_demographic.columns = ['patient_id', 'disease']
    return patient_demographic

def process_lung_data(patient_lung_data):
    lung_data = pd.DataFrame(patient_lung_data)
    lung_data.columns = ['patient_id', 'filename', 'start', 'end', 'crackles', 'wheezes']
    return lung_data

