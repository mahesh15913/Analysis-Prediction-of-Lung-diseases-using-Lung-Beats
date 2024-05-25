from data_preprocessing import read_data, parse_patient_lung_data, parse_metadata, merge_data, save_processed_data
from feature_extraction import extract_features, feature_padding
from model_definition import build_combined_model
from training_evaluation import prepare_data, encode_labels, train_model, plot_history
from prediction import make_predictions, evaluate_predictions
from utils import create_directories

import os

def main():
    # Directories
    root = "path_to_root_directory"
    processed_clips = os.path.join(root, "processed_clips")
    diagnosis_path = os.path.join(root, "patient_diagnosis.csv")
    create_directories([processed_clips])
    
    # Data Preprocessing
    content = read_data(processed_clips)
    lung_data = parse_patient_lung_data(processed_clips, content)
    metadata = parse_metadata(content)
    final_data = merge_data(metadata, lung_data, diagnosis_path)
    save_processed_data(final_data, 'processed_data.csv')
    
    # Feature Extraction
    mfcc, cstft, mSpec = extract_features(final_data, processed_clips)
    mfcc_padded = feature_padding(mfcc)
    cstft_padded = feature_padding(cstft)
    mSpec_padded = feature_padding(mSpec)
    
    # Model Definition
    input_shapes = {'mfcc': mfcc_padded.shape[1:], 'cstft': cstft_padded.shape[1:], 'mspec': mSpec_padded.shape[1:]}
    model = build_combined_model(input_shapes)
    
    # Training and Evaluation
    X_train, X_test, Y_train, Y_test = prepare_data(final_data)
    Y_train, Y_test, le = encode_labels(Y_train, Y_test)
    history = train_model(model, mfcc_padded, cstft_padded, mSpec_padded, Y_train, mfcc_padded, cstft_padded, mSpec_padded, Y_test)
    plot_history(history)
    
    # Prediction
    pred_labels = make_predictions(model, mfcc_padded, cstft_padded, mSpec_padded, Y_test)
    Y_test_inv, pred_labels_inv = evaluate_predictions(pred_labels, Y_test, le)
    
    print(Y_test_inv)
    print(pred_labels_inv)

if __name__ == "__main__":
    main()
