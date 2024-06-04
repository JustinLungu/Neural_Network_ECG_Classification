from visualization import Visualization
from preprocessing import Preprocessing
from save_load import Save, Load
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np




WINDOW_SIZE = 200
OVERLAP = 0.5 #0.5 = 50%
VALID_ANNOTATIONS = {'N', 'R'}
INVALID_ANNOTATIONS = {'~', '+', '|'}

DO_PREPROCESSING = False

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)


def split_norm(prep: Preprocessing, patient: Visualization, signal1_windows, signal2_windows, annotation_windows):
    # Split the data
    prep.split_data(signal1_windows, signal2_windows, annotation_windows, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Plotting each signal individually from train, validation and test
    patient.plot_1_signal(prep.signal1.train, prep.labels.train, title="Signal 1 Training Data", filename="s1_train.png")
    patient.plot_1_signal(prep.signal1.val, prep.labels.val, title="Signal 1 Validation Data", filename="s1_val.png")
    patient.plot_1_signal(prep.signal1.test, prep.labels.test, title="Signal 1 Test Data", filename="s1_test.png")

    patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 2 Training Data", filename="s2_train.png")
    patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 2 Validation Data", filename="s2_val.png")
    patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 2 Test Data", filename="s2_test.png")

    prep.normalize_minmax()

    patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 1 Training Data Normalized", filename="s1_train_norm.png")
    patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 1 Validation Data Normalized", filename="s1_val_norm.png")
    patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 1 Test Data Normalized", filename="s1_test_norm.png")

    patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 2 Training Data Normalized", filename="s2_train_norm.png")
    patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 2 Validation Data Normalized", filename="s2_val_norm.png")
    patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 2 Test Data Normalized", filename="s2_test_norm.png")

    annotation_counts = Counter(annotation_windows)
    print(annotation_counts)

if __name__ == "__main__":
    patient = Visualization(212)
    #p_212.multi_plot_label()
    #p_212.plot_annotation_sep_channels()
    #p_212.plot_annotation_signals()
    prep = Preprocessing(patient.record, patient.annotation, VALID_ANNOTATIONS, INVALID_ANNOTATIONS)

    # Example usage
    if DO_PREPROCESSING is True:
        
        data_windows, annotation_windows = prep.extract_windows(WINDOW_SIZE, OVERLAP)
        signal1_windows = data_windows[:, :, 0:1]
        signal2_windows = data_windows[:, :, 1:2]
        
        # !! Verify that only the valid classes have been saved in the annotation array
        annotation_counts = Counter(annotation_windows)
        print(annotation_counts)

        save = Save(212, signal1_windows, signal2_windows, annotation_windows)
        save.save_data_json()
        save.save_data_csv()
        
    else:
        load = Load(212)
        signal1_windows, signal2_windows, annotation_windows = load.load_data_json()
        signal1_windows_c, signal2_windows_c, annotation_windows_c = load.load_data_csv()

    
    split_norm(prep, patient, signal1_windows, signal2_windows, annotation_windows)

    