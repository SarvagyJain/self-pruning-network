import numpy as np
import pytest

pytest.importorskip("scipy.sparse")

from aqua_nn.models import MLP


def _mask_model(model, seed=7):
    rng = np.random.default_rng(seed)
    for layer, keep_probability in (
        (model.layer1, 0.65),
        (model.layer2, 0.45),
        (model.layer3, 0.70),
    ):
        mask = (rng.random(layer.mask.shape) < keep_probability).astype(float)
        layer.set_mask(mask)


def test_sparse_logits_match_dense_masked_logits():
    model = MLP(seed=42)
    _mask_model(model)
    x = np.random.default_rng(8).normal(size=(13, 64))

    dense_logits = model.predict_logits_dense(x)
    sparse_weights = model.to_sparse_weights()
    sparse_logits = model.predict_logits_sparse(x, sparse_weights)

    np.testing.assert_allclose(sparse_logits, dense_logits, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(
        np.argmax(sparse_logits, axis=1),
        np.argmax(dense_logits, axis=1),
    )


def test_sparse_logits_match_dense_without_pruning():
    model = MLP(seed=19)
    x = np.random.default_rng(20).normal(size=(5, 64))

    np.testing.assert_allclose(
        model.predict_logits_sparse(x),
        model.predict_logits_dense(x),
        rtol=1e-10,
        atol=1e-10,
    )


def test_sparse_inference_handles_fully_pruned_layer():
    model = MLP(seed=42)
    model.layer2.set_mask(np.zeros_like(model.layer2.mask))
    x = np.ones((4, 64))

    np.testing.assert_allclose(
        model.predict_logits_sparse(x),
        model.predict_logits_dense(x),
        rtol=1e-10,
        atol=1e-10,
    )


def test_sparse_inference_rejects_wrong_number_of_matrices():
    model = MLP(seed=42)
    with pytest.raises(ValueError, match="three SciPy sparse matrices"):
        model.predict_logits_sparse(np.ones((2, 64)), [])


def test_sparse_inference_rejects_wrong_input_width():
    model = MLP(seed=42)
    with pytest.raises(ValueError, match="x must have shape"):
        model.predict_logits_sparse(np.ones((2, 63)))


def test_sparse_inference_rejects_matrix_shape_mismatch():
    from scipy.sparse import csr_matrix

    model = MLP(seed=42)
    wrong = [
        csr_matrix(np.zeros((64, 128))),
        csr_matrix(np.zeros((64, 128))),
        csr_matrix(np.zeros((64, 10))),
    ]
    with pytest.raises(ValueError, match="Sparse weight shape"):
        model.predict_logits_sparse(np.ones((2, 64)), wrong)
