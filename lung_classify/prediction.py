import numpy as np

def make_predictions(model, mfcc_test, cstft_test, mSpec_test, Y_test):
    predictions = model.predict({'mfcc': mfcc_test, 'cstft': cstft_test, 'mspec': mSpec_test})
    pred_labels = np.argmax(predictions, axis=1)
    return pred_labels

def evaluate_predictions(pred_labels, Y_test, le):
    Y_test_inv = le.inverse_transform(Y_test)
    pred_labels_inv = le.inverse_transform(pred_labels)
    return Y_test_inv, pred_labels_inv

# Example usage:
# pred_labels = make_predictions(model, mfcc_test, cstft_test, mSpec_test, Y_test)
# Y_test_inv, pred_labels_inv = evaluate_predictions(pred_labels, Y_test, le)
