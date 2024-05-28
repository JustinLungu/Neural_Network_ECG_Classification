from visualization import Visualization
from preprocessing import Preprocessing
from save_load import Save, Load

WINDOW_SIZE = 200
OVERLAP = 0.5 #0.5 = 50%
VALID_ANNOTATIONS = {'N', 'R'}
INVALID_ANNOTATIONS = {'~'}

if __name__ == "__main__":
    p_212 = Visualization(212)
    #p_212.multi_plot_label()
    #p_212.plot_annotation_sep_channels()
    #p_212.plot_annotation_signals()

    # Example usage
    prep = Preprocessing(p_212.record, p_212.annotation, VALID_ANNOTATIONS, INVALID_ANNOTATIONS)
    data_windows, annotation_windows = prep.extract_windows(WINDOW_SIZE, OVERLAP)
    signal1_windows = data_windows[:, :, 0:1]
    signal2_windows = data_windows[:, :, 1:2]

    save = Save(212, signal1_windows, signal2_windows, annotation_windows)
    load = Load(212)

    save.save_data_json()
    save.save_data_csv()
    signal1_windows, signal2_windows, annotation_windows = load.load_data_json()
    signal1_windows_c, signal2_windows_c, annotation_windows_c = load.load_data_csv()

    print(signal1_windows.shape)
    print(signal2_windows.shape)
    print(annotation_windows.shape)

    print(signal1_windows_c.shape)
    print(signal2_windows_c.shape)
    print(annotation_windows_c.shape)
