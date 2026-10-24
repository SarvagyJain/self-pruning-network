"""Train pruning sweeps and report validation/test outcomes.

The experiment is configurable through config.json and command-line overrides.
Validation accuracy selects a checkpoint; the held-out test set is evaluated
once on that selected checkpoint and is never used to choose weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.checkpoint import load_model_checkpoint, save_model_checkpoint
from aqua_nn.data.dataset import load_dataset_splits
from aqua_nn.losses import accuracy, softmax_cross_entropy
from aqua_nn.models import MLP
from aqua_nn.optim import Adam
from aqua_nn.pruning import (
    cubic_sparsity,
    global_prune,
    layerwise_prune,
    magnitude_scores,
    taylor_scores,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "raw"
DEFAULT_CONFIG = {
    "dataset": "digits",
    "seeds": [42],
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "target_sparsities": [0.0, 0.30, 0.50, 0.70, 0.90],
    "method": "gradual",
    "scope": "global",
    "importance": "magnitude",
    "schedule": "cubic",
    "warmup_epochs": 5,
    "prune_at_epoch": 10,
    "run_name": "",
}


def load_config(path: Path) -> dict:
    """Load JSON config and fill any omitted values with documented defaults."""
    config = DEFAULT_CONFIG.copy()
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            supplied = json.load(file)
        if not isinstance(supplied, dict):
            raise ValueError("Configuration root must be a JSON object")
        config.update(supplied)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--dataset", choices=("digits", "fashion-mnist"))
    parser.add_argument("--seeds", help="Comma-separated random seeds, e.g. 42,43,44")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--target-sparsities", help="Comma-separated ratios, e.g. 0,0.5,0.9")
    parser.add_argument("--method", choices=("gradual", "one_shot"))
    parser.add_argument("--scope", choices=("global", "layerwise"))
    parser.add_argument("--importance", choices=("magnitude", "taylor"))
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--prune-at-epoch", type=int)
    parser.add_argument("--run-name", help="Optional label to keep experiment outputs separate")
    return parser.parse_args()


def apply_cli_overrides(config: dict, args) -> dict:
    for cli_name, config_name in (
        ("dataset", "dataset"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("method", "method"),
        ("scope", "scope"),
        ("importance", "importance"),
        ("warmup_epochs", "warmup_epochs"),
        ("prune_at_epoch", "prune_at_epoch"),
        ("run_name", "run_name"),
    ):
        value = getattr(args, cli_name, None)
        if value is not None:
            config[config_name] = value
    if args.seeds:
        config["seeds"] = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if args.target_sparsities:
        config["target_sparsities"] = [
            float(value.strip()) for value in args.target_sparsities.split(",") if value.strip()
        ]
    return config


def validate_config(config: dict) -> None:
    if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not config["seeds"]:
        raise ValueError("At least one seed is required")
    if not config["target_sparsities"]:
        raise ValueError("At least one target sparsity is required")
    for value in config["target_sparsities"]:
        if not 0.0 <= float(value) < 1.0:
            raise ValueError("Every target sparsity must be in [0, 1)")
    if config["method"] not in {"gradual", "one_shot"}:
        raise ValueError("method must be 'gradual' or 'one_shot'")
    if config["scope"] not in {"global", "layerwise"}:
        raise ValueError("scope must be 'global' or 'layerwise'")
    if config["importance"] not in {"magnitude", "taylor"}:
        raise ValueError("importance must be 'magnitude' or 'taylor'")
    if config["schedule"] != "cubic":
        raise ValueError("Only the cubic schedule is currently implemented")
    if config["method"] == "gradual" and not 0 <= int(config["warmup_epochs"]) < int(config["epochs"]):
        raise ValueError("warmup_epochs must be >= 0 and less than epochs for gradual pruning")
    if config["method"] == "one_shot" and not 1 <= int(config["prune_at_epoch"]) <= int(config["epochs"]):
        raise ValueError("prune_at_epoch must be between 1 and epochs for one-shot pruning")


def evaluate(model, X, y, batch_size=512):
    """Evaluate loss and accuracy in batches to support larger datasets."""
    total_loss = 0.0
    correct = 0
    total = len(X)
    for start in range(0, total, batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(Tensor(xb))
        loss = softmax_cross_entropy(logits, yb).data.item()
        total_loss += float(loss) * len(xb)
        correct += int(np.sum(np.argmax(logits.data, axis=1) == yb))
    return {
        "loss": float(total_loss / total) if total else 0.0,
        "accuracy": float(correct / total) if total else 0.0,
    }


def prunable_layers(model):
    return [model.layer1, model.layer2, model.layer3]


def measure_sparsity(model):
    layers = prunable_layers(model)
    total_weights = sum(layer.weight.data.size for layer in layers)
    nonzero_weights = sum(int(np.count_nonzero(layer.mask)) for layer in layers)
    sparsity = 1.0 - nonzero_weights / total_weights if total_weights else 0.0
    total_parameters = sum(parameter.data.size for parameter in model.parameters())
    return {
        "total_weights": int(total_weights),
        "nonzero_weights": int(nonzero_weights),
        "weight_sparsity": float(sparsity),
        "total_parameters": int(total_parameters),
    }


def apply_pruning(model, target_sparsity, scope="global", importance="magnitude"):
    layers = prunable_layers(model)
    weights_and_masks = [(layer.weight.data, layer.mask) for layer in layers]
    if importance == "magnitude":
        scores = [magnitude_scores(layer.weight.data) for layer in layers]
    elif importance == "taylor":
        scores = [
            taylor_scores(layer.weight.data, layer.weight.grad)
            for layer in layers
        ]
    else:
        raise ValueError(f"Unknown importance method: {importance}")

    pruner = global_prune if scope == "global" else layerwise_prune
    new_masks, achieved_sparsity = pruner(
        weights_and_masks, scores, target_sparsity
    )
    for layer, new_mask in zip(layers, new_masks):
        layer.set_mask(new_mask)
    return float(achieved_sparsity)


def _target_label(target_sparsity):
    return f"{target_sparsity:.2f}".replace(".", "p")


def _run_label(config):
    raw = config.get("run_name") or (
        f"{config['dataset']}_{config['method']}_{config['scope']}_{config['importance']}"
    )
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in str(raw))
    return cleaned.strip("_-") or "pruning_experiment"


def train_experiment(target_sparsity, seed, config, X_train, y_train, X_val, y_val, X_test, y_test):
    model = MLP(input_size=X_train.shape[1], seed=seed)
    optimizer = Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(seed)
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    steps_per_epoch = (len(X_train) + batch_size - 1) // batch_size
    warmup_step = int(config["warmup_epochs"]) * steps_per_epoch
    final_step = epochs * steps_per_epoch
    global_step = 0
    history = []
    best_val_accuracy = -1.0  # Best validation accuracy among target-sparsity checkpoints.
    best_any_epoch_val_accuracy = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    run_label = _run_label(config)
    checkpoint_dir = RESULTS_DIR / "checkpoints"
    checkpoint_path = checkpoint_dir / (
        f"{run_label}_seed{seed}_target{_target_label(target_sparsity)}.npz"
    )

    for epoch in range(epochs):
        indices = rng.permutation(len(X_train))
        total_loss = 0.0

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()
            logits = model(Tensor(x_batch))
            loss = softmax_cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            global_step += 1

            if target_sparsity > 0.0 and config["method"] == "gradual":
                scheduled_sparsity = cubic_sparsity(
                    global_step, float(target_sparsity), warmup_step, final_step
                )
                if scheduled_sparsity > 0.0:
                    apply_pruning(
                        model, scheduled_sparsity,
                        scope=config["scope"], importance=config["importance"],
                    )

            # One-shot mode prunes at the end of the configured epoch, then
            # continues training/fine-tuning the remaining epochs.
            last_batch = start + batch_size >= len(indices)
            if (
                target_sparsity > 0.0
                and config["method"] == "one_shot"
                and not getattr(model, "_one_shot_pruned", False)
                and epoch + 1 == int(config["prune_at_epoch"])
                and last_batch
            ):
                apply_pruning(
                    model, float(target_sparsity),
                    scope=config["scope"], importance=config["importance"],
                )
                model._one_shot_pruned = True

            total_loss += float(loss.data.item()) * len(batch_indices)

        train_loss = total_loss / len(X_train)
        validation = evaluate(model, X_val, y_val, batch_size=batch_size)
        sparsity_info = measure_sparsity(model)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "weight_sparsity": sparsity_info["weight_sparsity"],
        }
        history.append(row)

        print(
            f"Dataset={config['dataset']} Seed={seed} Target={target_sparsity:.0%} | "
            f"Epoch {epoch + 1:02d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={validation['loss']:.4f} | val_acc={validation['accuracy']:.4f} | "
            f"sparsity={sparsity_info['weight_sparsity']:.2%}"
        )

        best_any_epoch_val_accuracy = max(
            best_any_epoch_val_accuracy, float(validation["accuracy"])
        )
        # Only select checkpoints that actually reached the requested sparsity.
        # For gradually pruned runs this usually starts at the final schedule step;
        # for one-shot runs it starts after the prune-at epoch. This avoids reporting
        # the accuracy of a dense/partially pruned checkpoint as the target result.
        total_weight_count = sparsity_info["total_weights"]
        rounding_tolerance = (
            1.0 / total_weight_count if total_weight_count else 0.0
        ) + (1.5 / total_weight_count if total_weight_count else 0.0)
        target_reached = (
            float(target_sparsity) == 0.0
            or sparsity_info["weight_sparsity"] + rounding_tolerance
            >= float(target_sparsity)
        )
        if target_reached and validation["accuracy"] > best_val_accuracy:
            best_val_accuracy = float(validation["accuracy"])
            best_epoch = epoch + 1
            save_model_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    "dataset": config["dataset"],
                    "seed": seed,
                    "target_sparsity": float(target_sparsity),
                    "method": config["method"],
                    "scope": config["scope"],
                    "importance": config["importance"],
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_val_accuracy,
                    "best_any_epoch_val_accuracy": best_any_epoch_val_accuracy,
                    "validation_fraction": float(config["validation_fraction"]),
                    "test_fraction": float(config["test_fraction"]),
                },
            )

    elapsed_seconds = time.perf_counter() - start_time
    final_validation = evaluate(model, X_val, y_val, batch_size=batch_size)
    best_model, checkpoint_metadata = load_model_checkpoint(checkpoint_path)
    test_metrics = evaluate(best_model, X_test, y_test, batch_size=batch_size)
    sparsity_info = measure_sparsity(best_model)

    try:
        checkpoint_label = str(checkpoint_path.resolve().relative_to(ROOT))
    except ValueError:
        checkpoint_label = str(checkpoint_path.resolve())

    result = {
        "dataset": config["dataset"],
        "seed": int(seed),
        "method": config["method"],
        "scope": config["scope"],
        "importance": config["importance"],
        "target_sparsity": float(target_sparsity),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "best_any_epoch_val_accuracy": float(best_any_epoch_val_accuracy),
        "final_val_accuracy": float(final_validation["accuracy"]),
        "final_val_loss": float(final_validation["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "elapsed_seconds": round(elapsed_seconds, 3),
        **sparsity_info,
        "checkpoint_path": checkpoint_label,
    }
    return result, history


def _write_csv(rows, path):
    if not rows:
        raise ValueError("Cannot write an empty result set")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(rows):
    """Average repeated-seed runs per target for the existing plot/analyzer."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target_sparsity"])].append(row)

    summaries = []
    for (_, target), group in sorted(grouped.items()):
        summary = {
            "target_sparsity": float(target),
            "best_epoch": int(round(np.mean([r["best_epoch"] for r in group]))),
            "best_val_accuracy": float(np.mean([r["best_val_accuracy"] for r in group])),
            "final_val_accuracy": float(np.mean([r["final_val_accuracy"] for r in group])),
            "final_val_loss": float(np.mean([r["final_val_loss"] for r in group])),
            "elapsed_seconds": float(np.mean([r["elapsed_seconds"] for r in group])),
            "total_weights": int(round(np.mean([r["total_weights"] for r in group]))),
            "nonzero_weights": int(round(np.mean([r["nonzero_weights"] for r in group]))),
            "weight_sparsity": float(np.mean([r["weight_sparsity"] for r in group])),
            "total_parameters": int(round(np.mean([r["total_parameters"] for r in group]))),
            "test_accuracy": float(np.mean([r["test_accuracy"] for r in group])),
            "seed_count": len(group),
            "std_best_val_accuracy": float(np.std([r["best_val_accuracy"] for r in group])),
            "std_test_accuracy": float(np.std([r["test_accuracy"] for r in group])),
        }
        summaries.append(summary)
    return summaries


def main():
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    validate_config(config)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    histories = {}
    for seed in config["seeds"]:
        data = load_dataset_splits(
            dataset=config["dataset"],
            seed=int(seed),
            validation_fraction=float(config["validation_fraction"]),
            test_fraction=float(config["test_fraction"]),
        )
        X_train, y_train, X_val, y_val, X_test, y_test = data
        print(
            f"Loaded {config['dataset']}: train={len(X_train)}, "
            f"validation={len(X_val)}, test={len(X_test)}, features={X_train.shape[1]}"
        )

        for target_sparsity in config["target_sparsities"]:
            print("\n" + "=" * 72)
            print(
                f"Training: dataset={config['dataset']}, seed={seed}, "
                f"target sparsity={float(target_sparsity):.0%}, "
                f"method={config['method']}, scope={config['scope']}, "
                f"importance={config['importance']}"
            )
            print("=" * 72)
            result, history = train_experiment(
                float(target_sparsity), int(seed), config,
                X_train, y_train, X_val, y_val, X_test, y_test,
            )
            all_results.append(result)
            histories[f"{config['dataset']}|{seed}|{float(target_sparsity):.2f}"] = history
            print(
                f"Completed target={float(target_sparsity):.0%}: "
                f"best_val={result['best_val_accuracy']:.4f}, "
                f"test={result['test_accuracy']:.4f}, "
                f"sparsity={result['weight_sparsity']:.2%}"
            )

    summaries = aggregate_runs(all_results)
    experiment_dir = RESULTS_DIR / "experiments" / _run_label(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_csv = experiment_dir / "pruning_sweep_runs.csv"
    summary_csv = experiment_dir / "pruning_sweep.csv"
    report_json = experiment_dir / "pruning_sweep.json"

    # Keep one archive per configuration. Root-level files remain convenient
    # aliases for analysis and represent only the most recent configuration.
    _write_csv(all_results, run_csv)
    _write_csv(summaries, summary_csv)
    _write_csv(all_results, RESULTS_DIR / "pruning_sweep_runs.csv")
    _write_csv(summaries, RESULTS_DIR / "pruning_sweep.csv")
    report = {
        "run_name": _run_label(config),
        "settings": config,
        "runs": all_results,
        "summary_by_target": summaries,
        "training_history": histories,
        "notes": [
            "Checkpoints are selected using validation accuracy only after target sparsity is reached.",
            "Held-out test accuracy is measured only after loading the selected checkpoint.",
            "Multiple-seed standard deviations are descriptive and may be zero for one seed.",
            "Root-level result files point to the latest run; named configuration archives live under results/raw/experiments/.",
        ],
    }
    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with (RESULTS_DIR / "pruning_sweep.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nPruning sweep complete.")
    print(f"Run label: {_run_label(config)}")
    print(f"Per-run CSV: {run_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"JSON report: {report_json}")
    for row in summaries:
        print(
            f"target={row['target_sparsity']:.0%} | "
            f"best_val={row['best_val_accuracy']:.4f} ± {row['std_best_val_accuracy']:.4f} | "
            f"test={row['test_accuracy']:.4f} ± {row['std_test_accuracy']:.4f} | "
            f"sparsity={row['weight_sparsity']:.2%}"
        )


if __name__ == "__main__":
    main()
