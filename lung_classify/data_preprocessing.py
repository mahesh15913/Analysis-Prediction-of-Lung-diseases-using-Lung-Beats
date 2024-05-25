import pandas as pd
import os

def read_data(directory):
    content = os.listdir(directory)
    return content

def parse_patient_lung_data(directory, content):
    patient_lung_data = []
    for filename in content:
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split()
                    patient_id = filename.split('.')[0].split("_")[0]
                    recording_id = filename.split('.')[0]
                    patient_lung_data.append([patient_id, recording_id] + parts)
    return pd.DataFrame(patient_lung_data, columns=['patient_id', 'filename', 'start', 'end', 'crackles', 'wheezes'])

def parse_metadata(content):
    data = []
    for filename in content:
        if filename.endswith('.txt'):
            patient_id = filename.split('.')[0].split("_")[0]
            recording_index = filename.split('.')[0].split("_")[1]
            chest_location = filename.split('.')[0].split("_")[2]
            acquisition_mode = filename.split('.')[0].split("_")[3]
            recording_equipment = filename.split('.')[0].split("_")[4]
            data.append([patient_id, recording_index, chest_location, acquisition_mode, recording_equipment, filename])
    return pd.DataFrame(data, columns=['patient_id', 'recording_index', 'chest_location', 'acquisition_mode', 'recording_equipment', 'filename'])

def merge_data(metadata, lung_data, patient_diagnosis_path):
    patient_demographic = pd.read_csv(patient_diagnosis_path)
    patient_demographic.columns = ['patient_id', 'disease']
    data_merged = metadata.merge(lung_data, on=['patient_id', 'filename'], how='outer')
    data_merged = data_merged.drop_duplicates()
    final_data = data_merged.merge(patient_demographic, on='patient_id', how='outer')
    return final_data

def save_processed_data(data, path):
    data.to_csv(path, index=False)

# Example usage:
# directory = r"C:\Users\Asus\Desktop\lung_prediciton\Analysis-Prediction-of-Lung-diseases-using-Lung-Beats\Respiratory sounds\audio_and_txt_files"
# content = read_data(directory)
# lung_data = parse_patient_lung_data(directory, content)
# metadata = parse_metadata(content)
# final_data = merge_data(metadata, lung_data, r'C:\Users\Asus\Desktop\lung_prediciton\Analysis-Prediction-of-Lung-diseases-using-Lung-Beats\Respiratory sounds\patient_diagnosis.csv')
# save_processed_data(final_data, 'processed_data.csv')