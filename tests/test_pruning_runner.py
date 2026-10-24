import numpy as np

import run_pruning_experiment as runner
from aqua_nn.checkpoint import load_model_checkpoint


def test_small_pruning_run_saves_checkpoint_at_requested_sparsity(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    X_train, y_train, X_val, y_val, X_test, y_test = (
        runner.load_dataset_splits(dataset="digits", seed=9)
    )
    config = {
        "dataset": "digits",
        "epochs": 2,
        "batch_size": 512,
        "learning_rate": 0.001,
        "warmup_epochs": 0,
        "prune_at_epoch": 1,
        "method": "gradual",
        "scope": "global",
        "importance": "magnitude",
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
    }

    result, history = runner.train_experiment(
        0.5, 9, config,
        X_train, y_train, X_val, y_val, X_test, y_test,
    )

    checkpoint_path = tmp_path / "checkpoints" / "digits_gradual_global_magnitude_seed9_target0p50.npz"
    assert checkpoint_path.exists()
    model, metadata = load_model_checkpoint(checkpoint_path)
    measured_sparsity = 1.0 - (
        sum(int(np.count_nonzero(layer.mask)) for layer in
            (model.layer1, model.layer2, model.layer3))
        / sum(layer.mask.size for layer in (model.layer1, model.layer2, model.layer3))
    )
    assert measured_sparsity >= 0.5 - 1e-4
    assert result["test_accuracy"] >= 0.0
    assert result["best_val_accuracy"] >= 0.0
    assert metadata["target_sparsity"] == 0.5
    assert len(history) == 2


def test_config_validation_allows_irrelevant_one_shot_epoch_in_gradual_mode():
    config = runner.DEFAULT_CONFIG.copy()
    config.update({"epochs": 3, "warmup_epochs": 1, "method": "gradual", "prune_at_epoch": 10})
    runner.validate_config(config)
