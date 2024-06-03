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


def plot_multiple(values_list, annotations):
    """
    Plots the given arrays of values using matplotlib.

    Parameters:
    values_list (list of arrays): A list of numerical arrays to plot.
    annotations (list of str): Corresponding annotations for the values.
    """

    # Select 4 random indexes
    random_indexes = np.random.choice(len(values_list), 4, replace=False)

    # Get the values and annotations for the random indexes
    values_list = [values_list[idx] for idx in random_indexes]
    annotations = [annotations[idx] for idx in random_indexes]

    plt.figure(figsize=(12, 8))
    
    for i, (values, annotation) in enumerate(zip(values_list, annotations), 1):
        plt.subplot(2, 2, i)
        plt.plot(values, linestyle='-')
        plt.title(f"Annotation: {annotation}")
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p_212 = Visualization(212)
    #p_212.multi_plot_label()
    #p_212.plot_annotation_sep_channels()
    #p_212.plot_annotation_signals()

    # Example usage
    if DO_PREPROCESSING is True:
        prep = Preprocessing(p_212.record, p_212.annotation, VALID_ANNOTATIONS, INVALID_ANNOTATIONS)
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

    

    plot_multiple(signal1_windows, annotation_windows)
    plot_multiple(signal2_windows, annotation_windows)

    annotation_counts = Counter(annotation_windows)
    print(annotation_counts)
