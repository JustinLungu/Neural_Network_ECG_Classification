import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Preprocessing():

    def __init__(self, record, annotation, valid, invalid):
        self.record = record
        self.annotation = annotation
        self.valid_annotations = valid
        self.invalid_annotations = invalid

        self.signal1 = None
        self.signal2 = None
        self.labels = None

    def extract_windows(self, window_size, overlap):
        step_size = int(window_size * (1 - overlap))
        data_windows = []
        annotation_windows = []

        for start in range(0, len(self.record.p_signal) - window_size + 1, step_size):
            end = start + window_size
            window = self.record.p_signal[start:end]

            # Get the annotations in the current window
            annotations_in_window = self.annotation.sample[(self.annotation.sample >= start) & (self.annotation.sample < end)]
            annotation_symbols_in_window = [self.annotation.symbol[i] for i in range(len(self.annotation.sample)) if self.annotation.sample[i] in annotations_in_window]

            if len(annotation_symbols_in_window) == 0:
                continue

            # Check if any invalid annotations are in the window
            if any(ann in self.invalid_annotations for ann in annotation_symbols_in_window):
                continue

            # Determine the majority annotation
            annotation_counts = pd.Series(annotation_symbols_in_window).value_counts()
            majority_annotation = annotation_counts.idxmax()

            data_windows.append(window)
            annotation_windows.append(majority_annotation)

        return np.array(data_windows), np.array(annotation_windows)
    
    def split_data(self, signal1, signal2, labels, train_r, val_r, test_r, random_state=42):
        assert train_r + val_r + test_r == 1, "The sum of train_r, val_r, and test_r must be 1."

        # First split: training set and remaining set
        indices = np.arange(len(labels))
        train_indices, remaining_indices, train_labels, remaining_labels = train_test_split(
            indices, labels, test_size=(1 - train_r), random_state=random_state)

        # Second split: validation set and test set from the remaining set
        val_indices, test_indices, val_labels, test_labels = train_test_split(
            remaining_indices, remaining_labels, test_size=test_r/(val_r + test_r), random_state=random_state)

        # Split signal1 and signal2 using the indices
        train_signal1 = signal1[train_indices]
        val_signal1 = signal1[val_indices]
        test_signal1 = signal1[test_indices]

        train_signal2 = signal2[train_indices]
        val_signal2 = signal2[val_indices]
        test_signal2 = signal2[test_indices]


        self.signal1 = Data_Info(train_signal1, val_signal1, test_signal1)
        self.signal2 = Data_Info(train_signal2, val_signal2, test_signal2)
        self.labels = Data_Info(train_labels, val_labels, test_labels)
    
class Data_Info():
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test