import numpy as np
import pytest

from aqua_nn.data.dataset import load_dataset_splits, load_digits_splits


def test_digits_three_way_split_is_deterministic_and_disjoint_by_size():
    first = load_digits_splits(seed=11)
    second = load_digits_splits(seed=11)

    X_train, y_train, X_val, y_val, X_test, y_test = first
    assert len(X_train) + len(X_val) + len(X_test) == 1797
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 64
    assert len(X_val) > 0 and len(X_test) > 0
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)
    assert set(np.unique(y_train)) == set(range(10))
    assert set(np.unique(y_val)) == set(range(10))
    assert set(np.unique(y_test)) == set(range(10))


def test_dataset_loader_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset_splits(dataset="unknown")
