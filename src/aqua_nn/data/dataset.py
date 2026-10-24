"""Dataset loaders used by the self-pruning experiments."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split


def _stratified_three_way_split(X, y, seed, validation_fraction, test_fraction):
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must both be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=validation_fraction + test_fraction,
        random_state=seed,
        stratify=y,
    )
    test_share_of_holdout = test_fraction / (validation_fraction + test_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_share_of_holdout,
        random_state=seed + 1,
        stratify=y_holdout,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_digits_splits(
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, stratified train/validation/test splits of digits.

    Images are flattened to 64 features and normalized from 0..16 to 0..1.
    The validation set is for model selection; the test set should be evaluated
    only after selecting a checkpoint.
    """
    dataset = load_digits()
    X = dataset.data.astype(np.float32) / 16.0
    y = dataset.target.astype(np.int64)
    return _stratified_three_way_split(
        X, y, seed, validation_fraction, test_fraction
    )


def load_fashion_mnist_splits(
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download/cache Fashion-MNIST from OpenML and return stratified splits.

    The first call requires network access. The dataset is cached by scikit-learn.
    Inputs are flattened 28x28 grayscale images normalized to 0..1.
    """
    dataset = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = np.asarray(dataset.data, dtype=np.float32) / 255.0
    y = np.asarray(dataset.target, dtype=np.int64)
    return _stratified_three_way_split(
        X, y, seed, validation_fraction, test_fraction
    )


def load_dataset_splits(
    dataset: str = "digits",
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
):
    """Load a supported dataset and return train, validation, and test arrays."""
    dataset = dataset.strip().lower()
    if dataset == "digits":
        loader = load_digits_splits
    elif dataset in {"fashion-mnist", "fashion_mnist"}:
        loader = load_fashion_mnist_splits
    else:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Choose 'digits' or 'fashion-mnist'."
        )
    return loader(
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def load_smoke_dataset(
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the small digits dataset using the original train/validation API.

    Kept for backwards compatibility with the existing quick training workflow.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    dataset = load_digits()
    X = dataset.data.astype(np.float32) / 16.0
    y = dataset.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n_val = int(round(len(X) * validation_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]
