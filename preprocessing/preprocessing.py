import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import csv
from sklearn.model_selection import train_test_split

def plot_sig(ch1, ch2, title):
    plt.figure(figsize=(10, 4))
    plt.plot(ch1[1000:3000])
    plt.plot(ch2[1000:3000])
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()

#loading data
channel1=[]
channel2=[]
annotations=[]

with open("data212/ecg_signal1_212.csv", "r") as f1:
    reader = csv.reader(f1)
    next(reader)
    for row in reader:
        channel1.extend(map(float, row))

with open("data212/ecg_signal2_212.csv", "r") as f2:
    reader = csv.reader(f2)
    next(reader)
    for row in reader:
        channel2.extend(map(float, row))

with open("data212/ecg_annotations_212.csv", "r") as fa:
    reader = csv.reader(fa)
    next(reader)
    for row in reader:
        annotations.append(row[0])

data212 = list(zip(channel1, channel2, annotations))
data212 = np.array(data212)

#train/test split
train_size = int(0.7 * len(data212))
test_size = int(0.2 * len(data212))
locked_size = len(data212) - train_size - test_size

train_data, rest_of_data = train_test_split(data212, train_size=train_size, shuffle=False)
test_data, locked_test_data = train_test_split(rest_of_data, test_size=locked_size, shuffle=False)

#splitting back into channels
channel1_train, channel2_train, annotations_train = train_data[:, 0], train_data[:, 1], train_data[:, 2]
channel1_test, channel2_test, annotations_test = test_data[:, 0], test_data[:, 1], test_data[:, 2]
channel1_locked, channel2_locked, annotations_locked = locked_test_data[:, 0], locked_test_data[:, 1], locked_test_data[:, 2]

# Convert string elements to floats
channel1_train = [float(value) for value in channel1_train]
channel2_train = [float(value) for value in channel2_train]

channel1_test = [float(value) for value in channel1_test]
channel2_test = [float(value) for value in channel2_test]

channel1_locked = [float(value) for value in channel1_locked]
channel2_locked = [float(value) for value in channel2_locked]

print("Data is loaded.")
processed_list = []

'''NORMALIZATION + BASELINE FITTING: ADJUST FOR PREPROCESSING ON SPLIT DATASET???'''
plot_sig(channel1_train, channel2_train, "Original ECG signal")

#min-max normalization [-1,1] (results very small? consider sklearn.minmaxscaler)
min1 = np.min(channel1)
max1 = np.max(channel1)
min2 = np.min(channel2)
max2 = np.max(channel2)
ch1_norm = (channel1 - min1) / (max1 - min1)
ch2_norm = (channel2 - min2) / (max2 - min2)

plot_sig(ch1_norm, ch2_norm, "Normalized ECG signal")

'''
The medfilt function, short for "median filter," is a method used in signal processing to reduce noise in a signal. It replaces each data point with the median value in its neighboring window. This helps smooth out the signal while preserving sharp edges and important features.

Here's how medfilt works:

Windowing: It slides a window of a specified size (kernel) along the signal.
Sorting and Median Calculation: For each position of the window, it sorts the data points within the window and selects the middle value, which is the median.
Replacement: It replaces the value at the center of the window with the calculated median value.
Signal Characteristics: Consider the frequency content and variability of your signal. If your signal contains low-frequency baseline variations that you want to smooth out, a larger kernel size may be appropriate. However, 
if the baseline variations are high-frequency or rapid, a smaller kernel size might be sufficient.
'''
#https://www.kaggle.com/code/nelsonsharma/ecg-02-ecg-signal-pre-processing
#kernel size >> 101: signal to subtract was a straight line (maybe there is no neighbour window?)
# so there was no change to the baseline
#small kernel size just made the line worse
#but test this again later to see the exact effects

for channel in [ch1_norm, ch2_norm]:
  X0 = channel  # Read original signal
  X0 = medfilt(X0, 101)  # Apply median filter one by one on top of each other
  res = np.subtract(channel,X0)
  processed_list.append(res)

plot_sig(processed_list[0], processed_list[1], "Baseline fitted ECG signal")