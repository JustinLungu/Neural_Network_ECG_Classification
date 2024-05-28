# Neural_Network_ECG_Classification
Repository for the Neural Networks Project: Classifying pathological heartbeats from ECG signals

Useful links regarding data:
- https://en.ecgpedia.org/wiki/Basics
- https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm
- https://archive.physionet.org/physiobank/database/html/mitdbdir/records.htm
- https://physionet.org/content/mitdb/1.0.0/


**Annotations in ECG Data (N, R, L, F, V, etc.)**
/ - Paced Beat: Indicates a beat that was paced by an artificial pacemaker.

N - Normal Beat: Represents a normal heartbeat produced by natural pacemaking activity of the heart, specifically from the sinus node.

A - Atrial Premature Beat: A premature heartbeat originating from the atrial chambers of the heart.

f - Fusion of Paced and Normal Beat: This occurs when a natural beat fuses with a paced beat, showing characteristics of both.

V - Premature Ventricular Contraction (PVC): An early heartbeat originating from the ventricles, which is a common type of arrhythmia.

x - Non-conducted P-wave (Blocked APB): Represents an atrial beat that does not lead to a ventricular beat, often due to blockage in conduction pathways.

L - Left Bundle Branch Block Beat: This beat shows a delay or blockage in the pathway that sends electrical impulses to the left side of the heart.

R - Right Bundle Branch Block Beat: Similar to the L beat, but it affects the right side of the heart's conduction pathway.

F - Fusion of Ventricular and Normal Beat: This is a beat that shows properties of both a normal and a premature ventricular beat.

~ - Signal Quality Change: This symbol is used to denote a segment of the ECG where the signal quality changes, which could affect interpretation.

" - Rhythm Change: This symbol indicates a change in the rhythm of the heart, which could be due to various reasons such as arrhythmias or external factors affecting the heart's electrical activity.




**ECG Lead Placement (V1, V2, MLII, etc.)**

    MLII (Modified Lead II): This is one of the most commonly used leads in ECG recordings. It provides a view of the electrical activity of the heart from the right arm to the left leg, which is helpful for detecting arrhythmias and other cardiac abnormalities.
    V1, V2, V4, V5: These are part of the chest (precordial) leads. Each provides a different angle on the heart's electrical activity:
        V1: Located on the right side of the sternum in the fourth intercostal space.
        V2: Positioned directly opposite V1 on the left side of the sternum.
        V4: Positioned in the fifth intercostal space in the mid-clavicular line.
        V5: Located horizontally with V4 but in the anterior axillary line.


**Machine Learning task:** Arrhythmia Detection (CNN)

One of the most common applications is to develop models that can automatically detect different types of arrhythmias from ECG signals. This involves classifying each heartbeat into one of the several categories such as normal, atrial premature, ventricular premature, paced beat, etc. Machine learning models such as convolutional neural networks (CNNs) can be trained on segments of ECG signals labeled with these categories.
