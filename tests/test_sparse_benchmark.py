import numpy as np
from scipy.sparse import csr_matrix

from benchmark_sparse_inference import (
    batch_from_validation,
    current_rss_bytes,
    latency_ms,
    sparse_matrix_storage_bytes,
    traced_peak_bytes,
)


def test_batch_from_validation_repeats_to_requested_batch_size():
    X = np.arange(12, dtype=float).reshape(3, 4)
    batch = batch_from_validation(X, 8)
    assert batch.shape == (8, 4)
    np.testing.assert_array_equal(batch[:3], X)


def test_sparse_storage_counts_data_indices_and_indptr():
    matrix = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    expected = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert sparse_matrix_storage_bytes(matrix) == expected


def test_latency_and_traced_peak_helpers():
    fn = lambda x: x + 1.0
    median, p90 = latency_ms(fn, np.ones((2, 2)), warmup=1, repeats=3)
    peak = traced_peak_bytes(fn, np.ones((2, 2)))
    assert median >= 0.0
    assert p90 >= 0.0
    assert peak >= 0


def test_rss_helper_is_optional_but_never_negative():
    rss = current_rss_bytes()
    assert rss is None or rss >= 0
