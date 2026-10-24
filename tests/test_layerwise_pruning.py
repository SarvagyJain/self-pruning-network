import numpy as np

from aqua_nn.pruning import layerwise_prune, magnitude_scores


def test_layerwise_prune_hits_target_per_layer():
    weights1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    weights2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    masks = [np.ones_like(weights1), np.ones_like(weights2)]
    scores = [magnitude_scores(weights1), magnitude_scores(weights2)]

    new_masks, achieved = layerwise_prune(
        [(weights1, masks[0]), (weights2, masks[1])], scores, 0.5
    )

    assert achieved == 0.5
    assert int(new_masks[0].sum()) == 2
    assert int(new_masks[1].sum()) == 2


def test_layerwise_prune_never_revives_a_masked_weight():
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[1.0, 0.0], [1.0, 1.0]])

    new_masks, _ = layerwise_prune(
        [(weights, mask)], [magnitude_scores(weights)], 0.5
    )

    assert new_masks[0][0, 1] == 0.0
