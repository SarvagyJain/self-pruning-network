import numpy as np

from aqua_nn.checkpoint import load_model_checkpoint, save_model_checkpoint
from aqua_nn.models import MLP


def test_checkpoint_round_trip_preserves_weights_masks_and_metadata(tmp_path):
    model = MLP(seed=42)
    model.layer1.set_mask(
        (np.random.default_rng(1).random(model.layer1.mask.shape) > 0.25).astype(float)
    )
    model.layer3.set_mask(
        (np.random.default_rng(2).random(model.layer3.mask.shape) > 0.5).astype(float)
    )
    x = np.random.default_rng(3).normal(size=(7, 64))
    expected_logits = model.predict_logits_dense(x)

    path = save_model_checkpoint(
        model,
        tmp_path / "model.npz",
        metadata={"dataset": "digits", "seed": 42, "target_sparsity": 0.5},
    )
    restored, metadata = load_model_checkpoint(path)

    assert metadata["dataset"] == "digits"
    assert metadata["seed"] == 42
    np.testing.assert_allclose(restored.predict_logits_dense(x), expected_logits)
    for original, loaded in zip(
        (model.layer1, model.layer2, model.layer3),
        (restored.layer1, restored.layer2, restored.layer3),
    ):
        np.testing.assert_array_equal(loaded.mask, original.mask)
        np.testing.assert_allclose(loaded.weight.data, original.weight.data)


def test_checkpoint_can_reconstruct_non_default_input_dimension(tmp_path):
    model = MLP(input_size=784, seed=5)
    path = save_model_checkpoint(model, tmp_path / "fashion.npz", metadata={"dataset": "fashion-mnist"})
    restored, metadata = load_model_checkpoint(path)

    assert restored.layer1.weight.data.shape == (784, 128)
    assert metadata["dataset"] == "fashion-mnist"
    assert restored.predict_logits_dense(np.zeros((2, 784))).shape == (2, 10)
