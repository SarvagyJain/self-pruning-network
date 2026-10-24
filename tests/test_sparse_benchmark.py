import numpy as np
from scipy.sparse import csr_matrix

from benchmark_sparse_inference import (
    batch_from_validation,
    benchmark_checkpoint,
    current_rss_bytes,
    latency_ms,
    sparse_matrix_storage_bytes,
    traced_peak_bytes,
)
from aqua_nn.checkpoint import save_model_checkpoint
from aqua_nn.models import MLP


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



def test_benchmark_checkpoint_reports_matching_dense_and_sparse_paths(tmp_path):
    model = MLP(seed=42)
    rng = np.random.default_rng(4)
    for layer in (model.layer1, model.layer2, model.layer3):
        mask = (rng.random(layer.mask.shape) < 0.4).astype(float)
        layer.set_mask(mask)
    checkpoint = save_model_checkpoint(
        model,
        tmp_path / "benchmark_model.npz",
        metadata={
            "dataset": "digits",
            "seed": 42,
            "target_sparsity": 0.6,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
    )

    rows = benchmark_checkpoint(checkpoint, batch_sizes=[1, 8], warmup=1, repeats=2)

    assert len(rows) == 2
    assert all(row["logits_match_allclose"] for row in rows)
    assert all(row["predictions_match_exactly"] for row in rows)
    assert all(row["csr_weight_storage_bytes_including_indices"] > 0 for row in rows)
    assert all(row["dense_median_latency_ms"] >= 0 for row in rows)
