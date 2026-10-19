"""
Analyze self-pruning network experiment results.

Reads:
    results/raw/pruning_sweep.csv

Writes:
    results/figures/accuracy_vs_sparsity.png
    results/figures/active_weights_by_sparsity.png
    results/figures/pareto_frontier.png
    results/raw/pruning_pareto_frontier.csv

Dependencies:
    matplotlib
No pandas dependency is required.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "results" / "raw" / "pruning_sweep.csv"
RAW_DIR = ROOT / "results" / "raw"
FIGURES_DIR = ROOT / "results" / "figures"


def read_results(path: Path) -> list[dict]:
    """Read experiment results from CSV and convert numeric fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Run the pruning experiment first from the repository root."
        )

    numeric_fields = {
        "target_sparsity": float,
        "best_epoch": int,
        "best_val_accuracy": float,
        "final_val_accuracy": float,
        "final_val_loss": float,
        "elapsed_seconds": float,
        "total_weights": int,
        "nonzero_weights": int,
        "weight_sparsity": float,
        "total_parameters": int,
    }

    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"The CSV file is empty: {path}")

        for line_number, raw_row in enumerate(reader, start=2):
            row = {}
            for field, converter in numeric_fields.items():
                value = raw_row.get(field)
                if value is None or value.strip() == "":
                    raise ValueError(
                        f"Missing value for '{field}' on CSV line {line_number}."
                    )
                try:
                    row[field] = converter(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid value for '{field}' on CSV line {line_number}: "
                        f"{value!r}"
                    ) from exc
            rows.append(row)

    if not rows:
        raise ValueError(f"No experiment rows were found in {path}")

    return sorted(rows, key=lambda row: row["weight_sparsity"])


def calculate_pareto_frontier(rows: list[dict]) -> list[dict]:
    """
    Return nondominated rows when maximizing best validation accuracy
    and minimizing the number of active weights.
    """
    frontier = []
    for candidate in rows:
        dominated = any(
            other["nonzero_weights"] <= candidate["nonzero_weights"]
            and other["best_val_accuracy"] >= candidate["best_val_accuracy"]
            and (
                other["nonzero_weights"] < candidate["nonzero_weights"]
                or other["best_val_accuracy"] > candidate["best_val_accuracy"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)

    return sorted(frontier, key=lambda row: row["nonzero_weights"])


def save_pareto_csv(rows: list[dict], path: Path) -> None:
    """Save Pareto-frontier results to CSV."""
    fields = [
        "target_sparsity",
        "weight_sparsity",
        "best_epoch",
        "best_val_accuracy",
        "final_val_accuracy",
        "nonzero_weights",
        "total_weights",
        "total_parameters",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def save_accuracy_plot(rows: list[dict], path: Path) -> None:
    sparsity = [row["weight_sparsity"] * 100 for row in rows]
    best_accuracy = [row["best_val_accuracy"] * 100 for row in rows]
    final_accuracy = [row["final_val_accuracy"] * 100 for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(sparsity, best_accuracy, marker="o", label="Best validation accuracy")
    plt.plot(sparsity, final_accuracy, marker="s", label="Final validation accuracy")
    plt.xlabel("Actual weight sparsity (%)")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Validation Accuracy vs. Weight Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_active_weights_plot(rows: list[dict], path: Path) -> None:
    sparsity = [row["weight_sparsity"] * 100 for row in rows]
    active_weights = [row["nonzero_weights"] for row in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar([f"{value:.0f}%" for value in sparsity], active_weights)
    plt.xlabel("Actual weight sparsity")
    plt.ylabel("Active (nonzero) weights")
    plt.title("Active Weights Remaining After Pruning")
    plt.grid(axis="y", alpha=0.3)
    plt.bar_label(bars, fmt="%.0f", padding=3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_pareto_plot(
    rows: list[dict], frontier: list[dict], path: Path
) -> None:
    all_x = [row["nonzero_weights"] for row in rows]
    all_y = [row["best_val_accuracy"] * 100 for row in rows]
    frontier_x = [row["nonzero_weights"] for row in frontier]
    frontier_y = [row["best_val_accuracy"] * 100 for row in frontier]

    plt.figure(figsize=(8, 5))
    plt.scatter(all_x, all_y, label="All experiments")
    plt.plot(frontier_x, frontier_y, marker="o", label="Pareto frontier")
    for row in rows:
        label = f'{row["weight_sparsity"]:.0%} sparse'
        plt.annotate(
            label,
            (row["nonzero_weights"], row["best_val_accuracy"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Active (nonzero) weights — fewer is better")
    plt.ylabel("Best validation accuracy (%) — higher is better")
    plt.title("Accuracy vs. Active Weights: Pareto Frontier")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_results(INPUT_CSV)
    frontier = calculate_pareto_frontier(rows)

    save_accuracy_plot(rows, FIGURES_DIR / "accuracy_vs_sparsity.png")
    save_active_weights_plot(rows, FIGURES_DIR / "active_weights_by_sparsity.png")
    save_pareto_plot(rows, frontier, FIGURES_DIR / "pareto_frontier.png")
    save_pareto_csv(frontier, RAW_DIR / "pruning_pareto_frontier.csv")

    summary = {
        "experiment_count": len(rows),
        "pareto_frontier_count": len(frontier),
        "pareto_frontier": [
            {
                "sparsity_percent": round(row["weight_sparsity"] * 100, 2),
                "best_validation_accuracy_percent": round(
                    row["best_val_accuracy"] * 100, 3
                ),
                "active_weights": row["nonzero_weights"],
                "total_weights": row["total_weights"],
            }
            for row in frontier
        ],
        "generated_files": [
            str(FIGURES_DIR / "accuracy_vs_sparsity.png"),
            str(FIGURES_DIR / "active_weights_by_sparsity.png"),
            str(FIGURES_DIR / "pareto_frontier.png"),
            str(RAW_DIR / "pruning_pareto_frontier.csv"),
        ],
        "note": (
            "Weight masks report sparsity, but do not by themselves prove "
            "faster inference or lower allocated model memory."
        ),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
