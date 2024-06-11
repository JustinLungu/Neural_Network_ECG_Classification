import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from scipy.signal import medfilt, butter, filtfilt, iirnotch

class Preprocessing():

    
    def __init__(self, record, annotation, valid, invalid, artifacts):
        self.record = record
        self.annotation = annotation
        self.valid_annotations = valid
        self.invalid_annotations = invalid
        self.artifacts = artifacts



        self.signal1 = self.record.p_signal[:, 0].flatten()
        self.signal2 = self.record.p_signal[:, 1].flatten()
        #self.labels = None

    def is_artifact(self, start, end):
        """Check if the window falls within any of the intervals to remove."""
        for art_start, art_end in self.artifacts:
            if start < art_end and end > art_start:
                return True
        return False


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

            # Skip windows that fall within intervals to remove
            if self.is_artifact(start, end):
                continue

            # Determine the majority annotation
            annotation_counts = pd.Series(annotation_symbols_in_window).value_counts()
            majority_annotation = annotation_counts.idxmax()

            data_windows.append(window)
            annotation_windows.append(majority_annotation)

        return np.array(data_windows), np.array(annotation_windows)

    def butterworth(self, fs, lowcut, highcut):
        
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(5, [low, high], btype='band')
        self.signal1 = filtfilt(b, a, self.signal1)
        self.signal2 = filtfilt(b, a, self.signal2)

    def baseline_fitting(self):
        processed_list = []
        for channel in [self.signal1, self.signal2]:
            X0 = channel
            X0 = medfilt(X0, 101)
            res = np.subtract(channel, X0)
            processed_list.append(res)
        self.signal1 = processed_list[0]
        self.signal2 = processed_list[1]

    def remove_artifacts(self):
        pass

    def remove_high_low(self):
        pass

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

        # check the distribution of classes in each subset
        print("Distribution in training set:", Counter(train_labels))
        print("Distribution in validation set:", Counter(val_labels))
        print("Distribution in test set:", Counter(test_labels))

    def balance_data(self):
        smote = SMOTE(random_state=42)

        print("Before balancing:")
        print(Counter(self.labels.train))

        #reshape from 3D (samples, window, channels) to 2D (samples, window * channels)
        n_samples, window_size, n_channels = self.signal1.train.shape
        signal1_2d = self.signal1.train.reshape(n_samples, -1)
        signal2_2d = self.signal2.train.reshape(n_samples, -1)

        #apply smote: it expects data that has shape <= 2
        signal1_balanced, labels_balanced = smote.fit_resample(signal1_2d, self.labels.train)
        signal2_balanced, _ = smote.fit_resample(signal2_2d, self.labels.train)

        print("After balancing:")
        print(Counter(labels_balanced)) 

        # eeshape signals back to normal
        self.signal1.train = signal1_balanced.reshape(-1, window_size, n_channels)
        self.signal2.train = signal2_balanced.reshape(-1, window_size, n_channels)
        self.labels.train = labels_balanced


    def normalize_minmax(self):
        '''
        1. Identify the min and max
        2. apply the transformation: (x - min) / (max - min)
        3. we get values between 1 and 0
        '''
        scaler_signal1 = MinMaxScaler()
        scaler_signal2 = MinMaxScaler()

        self.signal1.train = scaler_signal1.fit_transform(self.signal1.train.reshape(-1, self.signal1.train.shape[-1])).reshape(self.signal1.train.shape)
        self.signal1.val = scaler_signal1.transform(self.signal1.val.reshape(-1, self.signal1.val.shape[-1])).reshape(self.signal1.val.shape)
        self.signal1.test = scaler_signal1.transform(self.signal1.test.reshape(-1, self.signal1.test.shape[-1])).reshape(self.signal1.test.shape)

        self.signal2.train = scaler_signal2.fit_transform(self.signal2.train.reshape(-1, self.signal2.train.shape[-1])).reshape(self.signal2.train.shape)
        self.signal2.val = scaler_signal2.transform(self.signal2.val.reshape(-1, self.signal2.val.shape[-1])).reshape(self.signal2.val.shape)
        self.signal2.test = scaler_signal2.transform(self.signal2.test.reshape(-1, self.signal2.test.shape[-1])).reshape(self.signal2.test.shape)

    
class Data_Info():
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test