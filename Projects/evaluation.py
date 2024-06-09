from __future__ import annotations

import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score


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

    def calculate_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average="binary")
        print(f"F1 Score: {f1:.4f}")
        return f1

    def plot_confusion_matrix(self, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.plots_path, "confusion_matrix.png"))
        plt.show()
