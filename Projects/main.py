from visualization import Visualization
from preprocessing import Preprocessing
from save_load import Save, Load, Model_Save_Load
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from model import CNNModel


# DATA SPLIT
# DATA NORMALIZATION
# DATA BALANCING

PATIENT_NUMBER = 212

WINDOW_SIZE = 200
OVERLAP = 0.5 #0.5 = 50%
VALID_ANNOTATIONS = {'N', 'R'}
NUM_CLASSES = len(VALID_ANNOTATIONS)
INVALID_ANNOTATIONS = {'~', '+', '|'}

DO_PREPROCESSING = False
DO_TRAINING = True

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

MODEL_NAME = "model"
EPOCHS = 20
BATCH_SIZE = 32
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
ACTIVATION = 'sigmoid'

def split_balance_norm(prep: Preprocessing, patient: Visualization, signal1_windows, signal2_windows, annotation_windows):
    # Split the data
    prep.split_data(signal1_windows, signal2_windows, annotation_windows, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Plotting each signal individually from train, validation and test
    
    # patient.plot_1_signal(prep.signal1.train, prep.labels.train, title="Signal 1 Training Data", filename="s1_train.png")
    # patient.plot_1_signal(prep.signal1.val, prep.labels.val, title="Signal 1 Validation Data", filename="s1_val.png")
    # patient.plot_1_signal(prep.signal1.test, prep.labels.test, title="Signal 1 Test Data", filename="s1_test.png")

    # patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 2 Training Data", filename="s2_train.png")
    # patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 2 Validation Data", filename="s2_val.png")
    # patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 2 Test Data", filename="s2_test.png")

    prep.balance_data()

    prep.normalize_minmax()

    # patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 1 Training Data Normalized", filename="s1_train_norm.png")
    # patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 1 Validation Data Normalized", filename="s1_val_norm.png")
    # patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 1 Test Data Normalized", filename="s1_test_norm.png")

    # patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 2 Training Data Normalized", filename="s2_train_norm.png")
    # patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 2 Validation Data Normalized", filename="s2_val_norm.png")
    # patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 2 Test Data Normalized", filename="s2_test_norm.png")

    

if __name__ == "__main__":
    patient = Visualization(PATIENT_NUMBER)
    #p_212.multi_plot_label()
    #p_212.plot_annotation_sep_channels()
    #p_212.plot_annotation_signals()
    prep = Preprocessing(patient.record, patient.annotation, VALID_ANNOTATIONS, INVALID_ANNOTATIONS, WINDOW_SIZE)

    # Example usage
    if DO_PREPROCESSING is True:
        
        data_windows, annotation_windows = prep.extract_windows(WINDOW_SIZE, OVERLAP)
        signal1_windows = data_windows[:, :, 0:1]
        signal2_windows = data_windows[:, :, 1:2]
        
        # !! Verify that only the valid classes have been saved in the annotation array
        annotation_counts = Counter(annotation_windows)
        print(annotation_counts)

        save = Save(PATIENT_NUMBER, signal1_windows, signal2_windows, annotation_windows)
        save.save_data_json()
        save.save_data_csv()
        
    else:
        load = Load(PATIENT_NUMBER)
        signal1_windows, signal2_windows, annotation_windows = load.load_data_json()
        signal1_windows_c, signal2_windows_c, annotation_windows_c = load.load_data_csv()

    
    split_balance_norm(prep, patient, signal1_windows, signal2_windows, annotation_windows)
    

    #TODO: also don't forget to do any other preprocessing needed

    
    
    ####################### Training the Model ########################################
    train_data = [prep.signal1.train, prep.signal2.train]
    val_data = [prep.signal1.val, prep.signal2.val]
    test_data = [prep.signal1.test, prep.signal2.test]

    # Convert labels to binary (0 and 1)
    label_mapping = {'N': 0, 'R': 1}

    train_labels = np.vectorize(label_mapping.get)(prep.labels.train)
    val_labels = np.vectorize(label_mapping.get)(prep.labels.val)
    test_labels = np.vectorize(label_mapping.get)(prep.labels.test)

    sl_model = Model_Save_Load(PATIENT_NUMBER, MODEL_NAME)

    if DO_TRAINING == True:
        # Initialize and train the model
        cnn_model = CNNModel(train_data[0].shape[1:], train_data[1].shape[1:], NUM_CLASSES, ACTIVATION, OPTIMIZER, LOSS)
        cnn_model.train(train_data, train_labels, val_data, val_labels, EPOCHS, BATCH_SIZE)

        #saving the model
        sl_model.save_model_h5(cnn_model.model)
        sl_model.save_model_pkl(cnn_model.model)
        #WARNING: Does not work with tensorflow 2.16 but does work with 2.15
        #sl_model.save_model_tflite(cnn_model.model)
        

    else:
        #lod model
        cnn_model = sl_model.load_model_h5()

    # Evaluate the model
    test_loss, test_accuracy = cnn_model.evaluate(test_data, test_labels)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    
