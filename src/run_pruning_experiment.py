import csv
import json
import time
from pathlib import Path

import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.data.dataset import load_smoke_dataset
from aqua_nn.losses import accuracy, softmax_cross_entropy
from aqua_nn.models import MLP
from aqua_nn.optim import Adam
from aqua_nn.pruning import cubic_sparsity, global_prune, magnitude_scores


RESULTS_DIR = Path("results/raw")
SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TARGET_SPARSITIES = (0.0, 0.30, 0.50, 0.70, 0.90)
WARMUP_EPOCHS = 5


def evaluate(model, X, y):
    logits = model(Tensor(X))
    loss = softmax_cross_entropy(logits, y).data.item()
    return {"loss": float(loss), "accuracy": float(accuracy(logits, y))}


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


def apply_global_pruning(model, target_sparsity):
    layers = prunable_layers(model)
    weights_and_masks = [(layer.weight.data, layer.mask) for layer in layers]
    scores = [magnitude_scores(layer.weight.data) for layer in layers]
    new_masks, achieved_sparsity = global_prune(
        weights_and_masks, scores, target_sparsity
    )
    for layer, new_mask in zip(layers, new_masks):
        layer.set_mask(new_mask)
    return float(achieved_sparsity)


def train_experiment(target_sparsity, X_train, y_train, X_val, y_val):
    model = MLP(seed=SEED)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED)
    steps_per_epoch = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    warmup_step = WARMUP_EPOCHS * steps_per_epoch
    final_step = EPOCHS * steps_per_epoch
    global_step = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        indices = rng.permutation(len(X_train))
        total_loss = 0.0

        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]
            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()
            logits = model(Tensor(x_batch))
            loss = softmax_cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            global_step += 1

            if target_sparsity > 0.0:
                scheduled_sparsity = cubic_sparsity(
                    global_step, target_sparsity, warmup_step, final_step
                )
                if scheduled_sparsity > 0.0:
                    apply_global_pruning(model, scheduled_sparsity)

            total_loss += loss.data.item() * len(batch_indices)

        train_loss = total_loss / len(X_train)
        validation = evaluate(model, X_val, y_val)
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
            f"Target={target_sparsity:.0%} | Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={validation['loss']:.4f} | "
            f"val_acc={validation['accuracy']:.4f} | "
            f"sparsity={sparsity_info['weight_sparsity']:.2%}"
        )

    elapsed_seconds = time.perf_counter() - start_time
    final_validation = evaluate(model, X_val, y_val)
    sparsity_info = measure_sparsity(model)
    best_epoch_row = max(history, key=lambda row: row["val_accuracy"])

    result = {
        "target_sparsity": float(target_sparsity),
        "best_epoch": int(best_epoch_row["epoch"]),
        "best_val_accuracy": float(best_epoch_row["val_accuracy"]),
        "final_val_accuracy": final_validation["accuracy"],
        "final_val_loss": final_validation["loss"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        **sparsity_info,
    }
    return result, history


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_val, y_val = load_smoke_dataset(seed=SEED)
    results = []
    histories = {}

    for target_sparsity in TARGET_SPARSITIES:
        print("\n" + "=" * 72)
        print(f"Training model with target sparsity {target_sparsity:.0%}")
        print("=" * 72)
        result, history = train_experiment(
            target_sparsity, X_train, y_train, X_val, y_val
        )
        results.append(result)
        histories[f"{target_sparsity:.2f}"] = history
        print(
            f"Completed {target_sparsity:.0%}: "
            f"best accuracy={result['best_val_accuracy']:.4f}, "
            f"final accuracy={result['final_val_accuracy']:.4f}, "
            f"actual sparsity={result['weight_sparsity']:.2%}"
        )

    csv_path = RESULTS_DIR / "pruning_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = RESULTS_DIR / "pruning_sweep.json"
    report = {
        "experiment": "gradual_global_magnitude_pruning",
        "dataset": "sklearn_digits_8x8",
        "seed": SEED,
        "architecture": [64, 128, 64, 10],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "importance": "absolute_weight_magnitude",
        "schedule": "cubic",
        "results": results,
        "training_history": histories,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nPruning sweep complete.")
    print(f"CSV report: {csv_path.resolve()}")
    print(f"JSON report: {json_path.resolve()}")
    print("\nSummary")
    for result in results:
        print(
            f"target={result['target_sparsity']:.0%} | "
            f"actual={result['weight_sparsity']:.2%} | "
            f"best_val_acc={result['best_val_accuracy']:.4f} | "
            f"final_val_acc={result['final_val_accuracy']:.4f} | "
            f"nonzero_weights={result['nonzero_weights']}/"
            f"{result['total_weights']}"
        )


if __name__ == "__main__":
    main()
