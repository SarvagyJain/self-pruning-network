"""Small deterministic dataset loader for development smoke tests.

Scikit-learn is used only to load the dataset. Model training,
autodiff, optimization, and pruning will be implemented by us.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits


def load_smoke_dataset(
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, normalize, shuffle and split the 8x8 digits dataset.

    Returns:
        X_train: float32 array with shape (n_train, 64)
        y_train: int64 array with shape (n_train,)
        X_val: float32 array with shape (n_val, 64)
        y_val: int64 array with shape (n_val,)
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    dataset = load_digits()

    # Pixels are originally in the range 0..16.
    X = dataset.data.astype(np.float32) / 16.0
    y = dataset.target.astype(np.int64)

    # Explicit generator makes the split reproducible.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_val = int(round(len(X) * validation_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val
