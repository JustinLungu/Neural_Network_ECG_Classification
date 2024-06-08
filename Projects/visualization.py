import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os


class Visualization():

    def __init__(self, p_number, plot_artifacts):
        self.p_number = p_number
        self.data_path = '../Neural_Network_ECG_Classification/Collected_Data/patient_' + str(p_number) + '/'
        self.plots_path = '../Neural_Network_ECG_Classification/Projects/Plots/'
        #self.data_path = '../Collected_Data/patient_' + str(p_number) + '/'
        #self.plots_path = '../Projects/Plots/'
        self.valid_annotations = {'N', 'A', 'V', 'f', 'x', 'L', 'R', 'F', '/', '~', 'Q', 'j', 'a', 'J', '!', 'E', 'S', '"', 'e'}
        self.record = wfdb.rdrecord(self.data_path + str(self.p_number))
        self.annotation = wfdb.rdann(self.data_path + str(self.p_number), 'atr')
        self.plot_artifacts = plot_artifacts

    def _save_plot(self, filename):
        # Create the directory if it doesn't exist
        save_dir = '../Neural_Network_ECG_Classification/Projects/Plots/'
        #save_dir = '../Projects/Plots/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the current plot
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
        print(f"Plot saved as {filename} in {save_dir}")

    def search_artifacts(self):
        ''' This function was used
        in order to manually inspect the signal,
        splitting it into windows of 2000 samples
        to detect artifacts'''
        length = len(self.record.p_signal)
        print(length)
        start = 630000
        end = start + 2000
        fin = start + 60000
        while end <= fin:
            plt.figure(figsize=(10, 4))
            plt.plot(self.record.p_signal[start:end])
            plt.title(f"from {start} to {end}")
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.show()
            start = end
            end = end + 2000

    def plot_all(self):
        '''This function plots the entire signal'''
        plt.figure(figsize=(10, 4))
        plt.plot(self.record.p_signal)
        plt.title(f"Whole signal")
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        self._save_plot(f'patient_{self.p_number}_all.png')
        plt.show()

    def multi_plot_label(self):
        '''Plots 4 random 1000 sample windows or
        Plots the artifacts, depending on self.plot_artifacts'''

        plt.figure(figsize=(15, 10))

        if self.plot_artifacts==0:
            num_plots = 4
            start = [np.random.randint(0, self.record.p_signal.shape[0] - 1000) for _ in range(num_plots)]
            end = [start_index + 1000 for start_index in start]
            num_cols = 2
        else:
            num_plots = 6
            start = [231500, 252000, 375000, 502250, 572000, 580500]
            end = [start_index + 1000 for start_index in start]
            num_cols = 3

        for i in range(1, num_plots+1):
            plt.subplot(2, num_cols, i)

            # Plot both channels
            for channel in range(self.record.p_signal.shape[1]):
                plt.plot(self.record.p_signal[start[i-1]:end[i-1], channel], label=f'Channel {channel + 1}')

            # Filter and plot annotations within the segment for both channels
            valid_indices = (self.annotation.sample >= start[i-1]) & (self.annotation.sample < end[i-1])
            segment_ann_indices = self.annotation.sample[valid_indices] - start[i-1]
            segment_ann_symbols = np.array(self.annotation.symbol)[valid_indices]

            for idx, symbol in zip(segment_ann_indices, segment_ann_symbols):
                # Plot annotations on top of both channels, adjust 'y' for visibility if necessary
                y_offset = self.record.p_signal[idx + start[i-1], 0]  # Adjust according to the signal amplitude
                plt.plot(idx, y_offset, 'ro')  # Mark annotation on the first channel
                plt.text(idx, y_offset, symbol, color='red', fontsize=12)

            plt.title(f'ECG Segment from {start[i-1]} to {end[i-1]}')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend()

        plt.tight_layout()
        self._save_plot(f'patient_{self.p_number}_multi_plot_label.png')
        plt.show()


    def plot_annotation_signals(self):
        time_per_sample = 1 / self.record.fs if self.record.fs else 1 / 360  # Default sampling frequency

        plt.figure(figsize=(12, 6))
        ax = plt.gca()  # Get current axis

        # Map each annotation type to a unique color
        unique_symbols = sorted(set(self.annotation.symbol).intersection(self.valid_annotations))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_symbols)))  # Viridis color map
        symbol_to_color = {symbol: color for symbol, color in zip(unique_symbols, colors)}

        # Plot annotations from both channels with the same color
        for idx, symbol in enumerate(self.annotation.symbol):
            if symbol in self.valid_annotations and idx < len(self.record.p_signal):
                time_point = self.annotation.sample[idx] * time_per_sample
                signal_values = self.record.p_signal[self.annotation.sample[idx]]

                for channel in range(self.record.p_signal.shape[1]):
                    plt.scatter(time_point, signal_values[channel], color=symbol_to_color[symbol], label=symbol if symbol not in ax.get_legend_handles_labels()[1] else "", alpha=0.6)

        plt.title(f'Annotation Signal Values for Patient {self.p_number}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend(title='Annotations')
        plt.grid(True)
        self._save_plot(f'patient_{self.p_number}_annotation_signals.png')
        plt.show()

    def plot_annotation_sep_channels(self):
        

        time_per_sample = 1 / self.record.fs if self.record.fs else 1 / 360  # Default sampling frequency

        # Setting up the plot with two subplots, one for each channel
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
        axes = axes.flatten()  # Flatten in case we use more subplots later

        # Map each annotation type to a unique color
        unique_symbols = sorted(set(self.annotation.symbol).intersection(self.valid_annotations))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_symbols)))  # Viridis color map
        symbol_to_color = {symbol: color for symbol, color in zip(unique_symbols, colors)}

        # Iterate over each channel
        for channel in range(self.record.p_signal.shape[1]):
            ax = axes[channel]  # Select the subplot for the current channel
            for idx, symbol in enumerate(self.annotation.symbol):
                if symbol in self.valid_annotations and idx < len(self.record.p_signal):
                    time_point = self.annotation.sample[idx] * time_per_sample
                    signal_value = self.record.p_signal[self.annotation.sample[idx]][channel]
                    ax.scatter(time_point, signal_value, color=symbol_to_color[symbol], label=symbol if symbol not in ax.get_legend_handles_labels()[1] else "", alpha=0.6)

            ax.set_title(f'Channel {channel + 1} Annotation Signals for Patient {self.p_number}')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)
            ax.legend(title='Annotations')

        plt.tight_layout()
        self._save_plot(f'patient_{self.p_number}_annotation_sep_channels.png')
        plt.show()


    def plot_1_signal(self, values_list, annotations, title, filename=None):
        """
        Plots the given arrays of values using matplotlib.

        Parameters:
        values_list (list of arrays): A list of numerical arrays to plot.
        annotations (list of str): Corresponding annotations for the values.
        title (str): The title for the plot.
        filename (str, optional): The filename to save the plot. If None, the plot is not saved.
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
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the suptitle
        
        if filename:
            save_path = os.path.join(self.plots_path, filename)
            plt.savefig(save_path)
            print(f"Plot saved as {filename} in {self.plots_path}")
        plt.show()
