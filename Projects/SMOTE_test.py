from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

# imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.90, 0.10],
    flip_y=0,
    random_state=42,
)

# visualize
plt.figure(figsize=(10, 5))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.5)
plt.title("Imbalanced Dataset")
plt.legend()
plt.show()

print("Original dataset shape:", Counter(y))

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Resampled dataset shape:", Counter(y_resampled))

# visualize the balanced
plt.figure(figsize=(10, 5))
plt.scatter(
    X_resampled[y_resampled == 0][:, 0],
    X_resampled[y_resampled == 0][:, 1],
    label="Class 0",
    alpha=0.5,
)
plt.scatter(
    X_resampled[y_resampled == 1][:, 0],
    X_resampled[y_resampled == 1][:, 1],
    label="Class 1",
    alpha=0.5,
)
plt.title("Balanced Dataset After Applying SMOTE")
plt.legend()
plt.show()
