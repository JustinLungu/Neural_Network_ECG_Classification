import json
import pandas as pd
import os
import numpy as np
import tensorflow as tf

class Save():

    def __init__(self, p_number, signal1, signal2, annotation):
        self.p_number = str(p_number)
        self.data_path = '../Neural_Network_ECG_Classification/Collected_Data/patient_' + str(p_number) + '/'
        self.model_path = '../Neural_Network_ECG_Classification/Models/'
        self.signal1 = signal1
        self.signal2 = signal2
        self.annotation = annotation

        # Create the directory if it doesn't exist
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def save_data_json(self):

        # Convert numpy arrays to lists
        signal1_windows_list = self.signal1.tolist()
        signal2_windows_list = self.signal2.tolist()
        annotation_windows_list = self.annotation.tolist()

        # Save to JSON
        data_json = {
            "signal1_windows": signal1_windows_list,
            "signal2_windows": signal2_windows_list,
            "annotation_windows": annotation_windows_list
        }

        file = "ecg_data_" + self.p_number + ".json"

        json_file_path = os.path.join(self.data_path, file)
        with open(json_file_path, "w") as json_file:
            json.dump(data_json, json_file)
    
        print("Data saved to ecg_data.json")

    def save_data_csv(self):
        # Save to CSV
        # Flatten the signal windows to 2D arrays for CSV saving
        signal1_windows_flat = self.signal1.reshape(self.signal1.shape[0], -1)
        signal2_windows_flat = self.signal2.reshape(self.signal2.shape[0], -1)
        
        df_signal1 = pd.DataFrame(signal1_windows_flat)
        df_signal2 = pd.DataFrame(signal2_windows_flat)
        df_annotations = pd.DataFrame(self.annotation, columns=["annotation"])

        signal1 = "ecg_signal1_" + self.p_number + ".csv"
        signal2 = "ecg_signal2_" + self.p_number + ".csv"
        annoations = "ecg_annotations_" + self.p_number + ".csv"

        signal1_csv_path = os.path.join(self.data_path, signal1)
        signal2_csv_path = os.path.join(self.data_path, signal2)
        annotations_csv_path = os.path.join(self.data_path, annoations)

        df_signal1.to_csv(signal1_csv_path, index=False)
        df_signal2.to_csv(signal2_csv_path, index=False)
        df_annotations.to_csv(annotations_csv_path, index=False)
        
        print(f"Data saved to {signal1_csv_path}, {signal2_csv_path}, and {annotations_csv_path}")

    

class Model_Save_Load():
    def __init__(self, p_number, model_name):
        self.p_number = str(p_number)
        self.model_path = '../Neural_Network_ECG_Classification/Models/'
        self.model_name = model_name

        # Create the directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def save_model(self, model):
        model_save_path = os.path.join(self.model_path, self.model_name)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    def load_model(self):
        model_load_path = os.path.join(self.model_path, self.model_name)
        model = tf.keras.models.load_model(model_load_path)
        print(f"Model loaded from {model_load_path}")
        return model



class Load():

    def __init__(self, p_number):
        self.p_number = str(p_number)
        self.data_path = '../Neural_Network_ECG_Classification/Collected_Data/patient_' + self.p_number + '/'
        self.model_path = '../Neural_Network_ECG_Classification/Models/'

    def load_data_json(self):
        file = "ecg_data_" + self.p_number + ".json"
        json_file_path = os.path.join(self.data_path, file)
        with open(json_file_path, "r") as json_file:
            data_json = json.load(json_file)

        signal1_windows = np.array(data_json["signal1_windows"])
        signal2_windows = np.array(data_json["signal2_windows"])
        annotation_windows = np.array(data_json["annotation_windows"])

        print(f"Data loaded from {json_file_path}")
        return signal1_windows, signal2_windows, annotation_windows

    def load_data_csv(self):
        signal1_csv_path = os.path.join(self.data_path, "ecg_signal1_" + self.p_number + ".csv")
        signal2_csv_path = os.path.join(self.data_path, "ecg_signal2_" + self.p_number + ".csv")
        annotations_csv_path = os.path.join(self.data_path, "ecg_annotations_" + self.p_number + ".csv")

        df_signal1 = pd.read_csv(signal1_csv_path)
        df_signal2 = pd.read_csv(signal2_csv_path)
        df_annotations = pd.read_csv(annotations_csv_path)

        signal1_windows = df_signal1.values.reshape(-1, 200, 1)
        signal2_windows = df_signal2.values.reshape(-1, 200, 1)
        annotation_windows = df_annotations["annotation"].values

        print(f"Data loaded from {signal1_csv_path}, {signal2_csv_path}, and {annotations_csv_path}")
        return signal1_windows, signal2_windows, annotation_windows



    