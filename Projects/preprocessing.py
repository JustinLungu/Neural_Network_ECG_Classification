import numpy as np
import pandas as pd

class Preprocessing():

    def __init__(self, record, annotation, valid, invalid):
        self.record = record
        self.annotation = annotation
        self.valid_annotations = valid
        self.invalid_annotations = invalid

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