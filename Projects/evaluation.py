from __future__ import annotations

import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.utils import resample
import numpy as np


class Evaluation:
    def __init__(self, history):
        self.history = history
        self.plots_path = "../Neural_Network_ECG_Classification/Projects/Plots/"

    def plot_loss(self):

        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, "loss_plot.png"))
        plt.show()

    def plot_accuracy(self):

        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_path, "accuracy.png"))
        plt.show()

    def calculate_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average="binary")
        print(f"F1 Score: {f1:.4f}")
        return f1

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        # Find the minimum number of samples for any label
        unique_labels, counts = np.unique(y_true, return_counts=True)
        min_samples = min(counts)
        
        # Resample each label to have the same number of samples as the label with the fewest samples
        balanced_y_true = []
        balanced_y_pred = []
        for label in unique_labels:
            indices = np.where(y_true == label)[0]
            selected_indices = resample(indices, n_samples=min_samples, replace=False, random_state=42)
            balanced_y_true.extend(y_true[selected_indices])
            balanced_y_pred.extend(y_pred[selected_indices])
        
        balanced_y_true = np.array(balanced_y_true)
        balanced_y_pred = np.array(balanced_y_pred)

        # Calculate and plot confusion matrix
        cm = confusion_matrix(balanced_y_true, balanced_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.plots_path, "confusion_matrix.png"))
        plt.show()