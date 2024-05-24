import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from data_processing import load_data, parse_patient_lung_data, create_patient_dataframe, merge_dataframes, load_patient_demographic, process_lung_data
from feature_extraction import extract_clips, extract_features, feature_padding
from model_training import build_model, train_model, evaluate_model, plot_history

def main():
    directory = "./Analysis-Prediction-of-Lung-diseases-using-Lung-Beats/Respiratory sounds"
    content = load_data(directory)
    
    patient_lung_data = parse_patient_lung_data(content, directory)
    data = create_patient_dataframe(content)
    lung_data = process_lung_data(patient_lung_data)
    datay = merge_dataframes(data, lung_data)
    
    patient_demographic = load_patient_demographic(r'C:\Users\Asus\Desktop\lung_prediciton\Analysis-Prediction-of-Lung-diseases-using-Lung-Beats\Respiratory sounds\patient_diagnosis.csv')
    final_data = datay.merge(patient_demographic, how='outer', on='patient_id')
    
    X_train, X_test, Y_train, Y_test = train_test_split(final_data, final_data['disease'], stratify=final_data['disease'], random_state=43, test_size=0.25)
    
    save_dir = 'processed_clips/'
    extract_clips(X_train, directory, save_dir)
    extract_clips(X_test, directory, save_dir)
    
    mfcc_train, cstft_train, mSpec_train = extract_features(X_train, save_dir)
    mfcc_test, cstft_test, mSpec_test = extract_features(X_test, save_dir)
    
    mfcc_train = np.array(feature_padding(mfcc_train))
    cstft_train = np.array(feature_padding(cstft_train))
    mSpec_train = np.array(feature_padding(mSpec_train))
    mfcc_test = np.array(feature_padding(mfcc_test))
    cstft_test = np.array(feature_padding(cstft_test))
    mSpec_test = np.array(feature_padding(mSpec_test))
    
    model = build_model()
    history = train_model(model, mfcc_train, cstft_train, mSpec_train, Y_train, mfcc_test, cstft_test, mSpec_test, Y_test)
    
    evaluate_model(model, mfcc_test, cstft_test, mSpec_test, Y_test)
    plot_history(history)

if __name__ == "__main__":
    main()
