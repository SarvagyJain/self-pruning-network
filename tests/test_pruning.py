
import numpy as np

from aqua_nn.pruning import (
    magnitude_scores,
    taylor_scores,
    cubic_sparsity,
    global_prune,
)


def test_magnitude_scores():
    weights = np.array([[0.1, -2.0], [0.5, -0.2]])
    np.testing.assert_allclose(
        magnitude_scores(weights),
        [[0.1, 2.0], [0.5, 0.2]],
    )


def test_taylor_scores():
    weights = np.array([[2.0, 3.0]])
    gradients = np.array([[0.5, -0.1]])
    np.testing.assert_allclose(
        taylor_scores(weights, gradients),
        [[1.0, 0.3]],
    )


def test_cubic_schedule():
    assert cubic_sparsity(0, 0.8, 0, 10) == 0.0
    assert np.isclose(cubic_sparsity(5, 0.8, 0, 10), 0.7)
    assert cubic_sparsity(10, 0.8, 0, 10) == 0.8
    assert cubic_sparsity(20, 0.8, 0, 10) == 0.8


def test_global_prune_reaches_target():
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.ones_like(weights)
    scores = magnitude_scores(weights)

    masks, achieved = global_prune(
        [(weights, mask)], [scores], target_sparsity=0.5
    )

    assert achieved == 0.5
    np.testing.assert_array_equal(masks[0], [[0, 0], [1, 1]])


def test_global_prune_never_revives_masked_weights():
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])
    scores = magnitude_scores(weights)

    masks, _ = global_prune(
        [(weights, mask)], [scores], target_sparsity=0.5
    )

    assert masks[0][0, 1] == 0.0
