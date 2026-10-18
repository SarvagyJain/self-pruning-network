import numpy as np

from aqua_nn.data.dataset import load_smoke_dataset


def test_smoke_dataset_shapes_and_types():
    X_train, y_train, X_val, y_val = load_smoke_dataset(seed=42)

    assert X_train.shape == (1438, 64)
    assert y_train.shape == (1438,)
    assert X_val.shape == (359, 64)
    assert y_val.shape == (359,)

    assert X_train.dtype == np.float32
    assert X_val.dtype == np.float32
    assert y_train.dtype == np.int64
    assert y_val.dtype == np.int64


def test_smoke_dataset_values_are_valid():
    X_train, y_train, X_val, y_val = load_smoke_dataset()

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0
    assert set(np.unique(y_train)).issubset(set(range(10)))
    assert set(np.unique(y_val)).issubset(set(range(10)))


def test_smoke_dataset_is_reproducible():
    first = load_smoke_dataset(seed=123)
    second = load_smoke_dataset(seed=123)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(first_array, second_array)


def test_different_seeds_change_the_split():
    first = load_smoke_dataset(seed=42)
    second = load_smoke_dataset(seed=7)

    assert not np.array_equal(first[0], second[0])
    