
import csv
import json
import time
from pathlib import Path

import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.data.dataset import load_smoke_dataset
from aqua_nn.losses import softmax_cross_entropy, accuracy
from aqua_nn.models import MLP
from aqua_nn.optim import Adam


RESULTS_DIR = Path("results/raw")


def evaluate(model, X, y):
    """Calculate validation loss, accuracy and confusion matrix."""
    logits = model(Tensor(X))

    loss = softmax_cross_entropy(logits, y).data.item()
    acc = accuracy(logits, y)

    predictions = np.argmax(logits.data, axis=1)
    num_classes = logits.data.shape[1]

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )
    np.add.at(confusion, (y, predictions), 1)

    class_counts = confusion.sum(axis=1)
    class_accuracy = np.divide(
        confusion.diagonal(),
        class_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=class_counts != 0,
    )

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "confusion_matrix": confusion.tolist(),
        "class_accuracy": class_accuracy.tolist(),
    }


def save_model(model, path):
    """Save model weights and biases in a NumPy archive."""
    np.savez(
        path,
        layer1_weight=model.layer1.weight.data,
        layer1_bias=model.layer1.bias.data,
        layer2_weight=model.layer2.weight.data,
        layer2_bias=model.layer2.bias.data,
        layer3_weight=model.layer3.weight.data,
        layer3_bias=model.layer3.bias.data,
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seed = 42
    epochs = 30
    batch_size = 64
    learning_rate = 0.001

    X_train, y_train, X_val, y_val = load_smoke_dataset(
        seed=seed
    )

    model = MLP(seed=seed)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    rng = np.random.default_rng(seed)
    history = []

    best_accuracy = -1.0
    best_epoch = 0
    best_metrics = None

    start_time = time.perf_counter()

    for epoch in range(epochs):
        indices = rng.permutation(len(X_train))
        total_loss = 0.0

        # Training phase
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]

            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()

            logits = model(Tensor(x_batch))
            loss = softmax_cross_entropy(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.data.item() * len(batch_indices)
            )

        mean_train_loss = total_loss / len(X_train)

        # Validation phase
        val_metrics = evaluate(model, X_val, y_val)

        row = {
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train loss: {mean_train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.4f}"
        )

        # Preserve the model with the best validation accuracy.
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()

            save_model(
                model,
                RESULTS_DIR / "baseline_model_best.npz",
            )

    elapsed_seconds = time.perf_counter() - start_time

    # Save the final epoch's model separately.
    save_model(
        model,
        RESULTS_DIR / "baseline_model_final.npz",
    )

    # Save per-epoch training history.
    history_path = RESULTS_DIR / "baseline_history.csv"

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    # Evaluate the final-epoch model.
    final_metrics = evaluate(model, X_val, y_val)

    # Count trainable parameters, including biases.
    parameter_count = sum(
        parameter.data.size
        for parameter in model.parameters()
    )

    results = {
        "experiment": "baseline",
        "dataset": "sklearn_digits_8x8",
        "seed": seed,
        "architecture": [64, 128, 64, 10],
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "parameter_count": int(parameter_count),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "final_validation": final_metrics,
    }

    metrics_path = RESULTS_DIR / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nBaseline evaluation complete.")
    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation accuracy: "
        f"{best_metrics['accuracy']:.4f}"
    )
    print(
        "Final validation accuracy: "
        f"{final_metrics['accuracy']:.4f}"
    )
    print(f"Trainable parameters: {parameter_count}")
    print(f"Training time: {elapsed_seconds:.2f} seconds")
    print(f"Saved results to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
