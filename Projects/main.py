from __future__ import annotations

from collections import Counter

import numpy as np
from evaluation import Evaluation
from model import CNNModel
from preprocessing import Preprocessing
from save_load import Load
from save_load import Model_Save_Load
from save_load import Save
from visualization import Visualization

PATIENT_NUMBER = 212
WINDOW_SIZE = 200
OVERLAP = 0.5  # 0.5 = 50%
VALID_ANNOTATIONS = {"N", "R"}
NUM_CLASSES = len(VALID_ANNOTATIONS)
INVALID_ANNOTATIONS = {"~", "+", "|"}

DO_PREPROCESSING = False
DO_TRAINING = True

SAMPLING_RATE = 360
LOWCUT = 0.5
HIGHCUT = 40.0

# there are also some especially noisy sections, but they are long
# if we removed them, we will lose data
#######################################################################################
"""
it was recommended by the professor to not remove manually the artifacts unless we find
a way to automate the artifact removal as we would need to keep these artifacts in the
testing data if we are to deploy to real life this project.
"""
# 400000 - 406000
# 578300 - 581200
# ARTIFACTS = [[232000, 232600], [252250, 252700], [375600, 375750],
#             [502300, 503000], [572500, 572750], [580800, 581200]]
#######################################################################################
ARTIFACTS = []

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

MODEL_NAME = "tinyVGG"
EPOCHS = 10
BATCH_SIZE = 32
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"
ACTIVATION = "sigmoid"


def split_balance_norm(
    prep: Preprocessing,
    patient: Visualization,
    signal1_windows,
    signal2_windows,
    annotation_windows,
):
    # Split the data
    prep.split_data(
        signal1_windows,
        signal2_windows,
        annotation_windows,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
    )

    # Plotting each signal individually from train, validation and test

    # patient.plot_1_signal(prep.signal1.train, prep.labels.train, title="Signal 1 Training Data", filename="s1_train.png") # noqa
    # patient.plot_1_signal(prep.signal1.val, prep.labels.val, title="Signal 1 Validation Data", filename="s1_val.png") # noqa
    # patient.plot_1_signal(prep.signal1.test, prep.labels.test, title="Signal 1 Test Data", filename="s1_test.png") # noqa

    # patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 2 Training Data", filename="s2_train.png") # noqa
    # patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 2 Validation Data", filename="s2_val.png") # noqa
    # patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 2 Test Data", filename="s2_test.png") # noqa

    prep.balance_data()

    prep.normalize_minmax()

    # patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 1 Training Data Normalized", filename="s1_train_norm.png") # noqa
    # patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 1 Validation Data Normalized", filename="s1_val_norm.png") # noqa
    # patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 1 Test Data Normalized", filename="s1_test_norm.png") # noqa

    # patient.plot_1_signal(prep.signal2.train, prep.labels.train, title="Signal 2 Training Data Normalized", filename="s2_train_norm.png") # noqa
    # patient.plot_1_signal(prep.signal2.val, prep.labels.val, title="Signal 2 Validation Data Normalized", filename="s2_val_norm.png") # noqa
    # patient.plot_1_signal(prep.signal2.test, prep.labels.test, title="Signal 2 Test Data Normalized", filename="s2_test_norm.png") # noqa


if __name__ == "__main__":
    patient = Visualization(PATIENT_NUMBER, 1)
    # p_212.plot_annotation_sep_channels()
    # p_212.plot_annotation_signals()
    prep = Preprocessing(
        patient.record,
        patient.annotation,
        VALID_ANNOTATIONS,
        INVALID_ANNOTATIONS,
        ARTIFACTS,
    )

    # Example usage
    if DO_PREPROCESSING is True:
        # patient.plot_all()
        prep.butterworth(SAMPLING_RATE, LOWCUT, HIGHCUT)
        prep.baseline_fitting()

        # prep.record.p_signal = np.array([list(tup) for tup in zip(prep.signal1, prep.signal2)]) # noqa
        # patient.record.p_signal = np.array([list(tup) for tup in zip(prep.signal1, prep.signal2)]) # noqa
        # patient.multi_plot_label()
        # patient.plot_all()
        """
        Turns out removing very high/very low amplitudes is not that necessary after
        applying Butterworth filtering and baseling fitting so this was not implemented.
        """

        data_windows, annotation_windows = prep.extract_windows(WINDOW_SIZE, OVERLAP)
        signal1_windows = data_windows[:, :, 0:1]
        signal2_windows = data_windows[:, :, 1:2]

        # !! Verify that only the valid classes have been saved in the annotation array
        annotation_counts = Counter(annotation_windows)
        print(annotation_counts)

        save = Save(
            PATIENT_NUMBER,
            signal1_windows,
            signal2_windows,
            annotation_windows,
        )
        save.save_data_json()
        save.save_data_csv()

    else:
        load = Load(PATIENT_NUMBER)
        signal1_windows, signal2_windows, annotation_windows = load.load_data_json()
        signal1_windows_c, signal2_windows_c, annotation_windows_c = (
            load.load_data_csv()
        )

    split_balance_norm(
        prep,
        patient,
        signal1_windows,
        signal2_windows,
        annotation_windows,
    )

    # Training the Model
    train_data = [prep.signal1.train, prep.signal2.train]
    val_data = [prep.signal1.val, prep.signal2.val]
    test_data = [prep.signal1.test, prep.signal2.test]

    # Convert labels to binary (0 and 1)
    label_mapping = {"N": 0, "R": 1}

    train_labels = np.vectorize(label_mapping.get)(prep.labels.train)
    val_labels = np.vectorize(label_mapping.get)(prep.labels.val)
    test_labels = np.vectorize(label_mapping.get)(prep.labels.test)

    sl_model = Model_Save_Load(PATIENT_NUMBER, MODEL_NAME)

    if DO_TRAINING:
        # Initialize and train the model
        cnn_model = CNNModel(
            train_data[0].shape[1:],
            train_data[1].shape[1:],
            NUM_CLASSES,
            ACTIVATION,
            OPTIMIZER,
            LOSS,
        )
        model_history = cnn_model.train(
            train_data,
            train_labels,
            val_data,
            val_labels,
            EPOCHS,
            BATCH_SIZE,
        )

        cnn_model.model.summary()

        # saving the model
        sl_model.save_model_h5(cnn_model.model)
        sl_model.save_model_pkl(cnn_model.model)
        # WARNING: Does not work with tensorflow 2.16 but does work with 2.15
        #sl_model.save_model_tflite(cnn_model.model)


        evaluation = Evaluation(model_history)
        evaluation.plot_loss()
        evaluation.plot_accuracy()

    else:
        # load model
        evaluation = Evaluation(None)
        cnn_model = sl_model.load_model_h5()

    # Evaluate the model
    test_loss, test_accuracy = cnn_model.evaluate(test_data, test_labels)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Make predictions
    test_predictions = cnn_model.predict(test_data)
    #test_predictions = cnn_model.model.predict(test_data)
    test_predictions = np.round(test_predictions).astype(int)

    # Calculate F1 Score
    f1 = evaluation.calculate_f1_score(test_labels, test_predictions)

    # Plot confusion matrix
    evaluation.plot_confusion_matrix(test_labels, test_predictions, labels=["N", "R"])