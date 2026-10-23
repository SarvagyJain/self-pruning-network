# Self-Pruning Network

A small neural-network library built from scratch with Python and NumPy. The project implements a custom reverse-mode autodifferentiation engine, a trainable multilayer perceptron (MLP), Adam optimization, gradual magnitude-based pruning, and experiment analysis.

The goal is to study a central ML engineering trade-off: **can we remove many weight connections while retaining validation accuracy, and does that sparsity translate into real inference or storage benefits?**

## Current status

### Implemented

- [x] Python project setup and dependency management with `uv`
- [x] Deterministic loader for scikit-learn's 8×8 handwritten-digits dataset
- [x] Custom reverse-mode autodiff `Tensor` operations
- [x] Dense layers, ReLU, and an MLP with architecture `64 → 128 → 64 → 10`
- [x] Stable softmax cross-entropy and accuracy calculation
- [x] Adam optimizer, including mask-aware parameter updates
- [x] Magnitude and Taylor importance-score helpers
- [x] Cubic sparsity schedule and global pruning masks
- [x] Baseline training, validation metrics, and training-history output
- [x] Pruning sweep across 0%, 30%, 50%, 70%, and 90% target sparsity
- [x] Plot generation and Pareto-frontier analysis
- [x] Automated tests for tensors, layers/training, masking, the dataset, and pruning

### Pending

- [ ] Implement and validate sparse-aware inference using a sparse representation
- [ ] Assert dense-versus-sparse logit and prediction equivalence
- [ ] Benchmark dense and sparse inference latency under identical conditions
- [ ] Measure actual runtime memory use in addition to estimated weight-storage bytes
- [ ] Repeat benchmark runs and document environment, uncertainty, and limitations

**Important:** masks create zeros in dense weight arrays, but dense matrix multiplication still computes with dense-shaped arrays. Therefore, the current sparsity measurements do not, by themselves, demonstrate faster inference or lower allocated memory.

## Recorded experiment observations

The following values came from the local pruning sweep during development. Re-run the experiment to regenerate the outputs and verify them in your own environment.

| Target sparsity | Best validation accuracy | Active weights | Total weights |
|---:|---:|---:|---:|
| 0% | 98.33% | 17,024 | 17,024 |
| 30% | 98.33% | 11,917 | 17,024 |
| 50% | 98.05% | 8,512 | 17,024 |
| 70% | 98.33% | 5,107 | 17,024 |
| 90% | 97.77% | 1,702 | 17,024 |

The current experimental candidates are:

- **70% sparsity — primary candidate:** retained the highest observed best validation accuracy (98.33%) with 5,107 active weights.
- **90% sparsity — aggressive-compression candidate:** retained 97.77% best validation accuracy with 1,702 active weights.

These are validation results on a small digits dataset, not proof of generalization to larger datasets or production workloads. The 70% and 90% candidates must still be tested with sparse-aware inference before making speed or total-memory claims.

## Setup

Install/synchronize dependencies:

```powershell
uv sync
```

Run tests:

```powershell
uv run pytest -q
```

Train the baseline:

```powershell
uv run python src/train_baseline.py
```

Run the pruning sweep:

```powershell
uv run python src/run_pruning_experiment.py
```

Analyze the generated pruning results:

```powershell
uv run python src/analyze_pruning_results.py
```

Run commands from the repository root. The scripts write generated outputs under `results/raw` and `results/figures`; these generated outputs may be ignored by Git, so check `git status` before expecting them to appear on GitHub.

## Design principles

- Implement the tensor engine, neural-network layers, optimizer, training loop, and pruning logic ourselves.
- Use NumPy for numerical operations and scikit-learn for dataset loading only.
- Validate behavior with automated tests and reproducible seeds.
- Report measured accuracy and sparsity separately from measured latency and memory.
- Treat validation-set results as estimates, not as a guarantee of performance on unseen real-world data.

## Repository structure

```text
src/
  aqua_nn/
    autodiff/       # custom reverse-mode autodiff
    data/           # digits dataset loader
    layers.py       # dense layer and ReLU
    losses.py       # softmax cross-entropy and accuracy
    models.py       # MLP
    optim.py        # Adam optimizer
    pruning.py      # scores, schedule, global masks
  train_baseline.py
  run_pruning_experiment.py
  analyze_pruning_results.py
tests/              # automated checks
```

## Dataset note

The current experiments use scikit-learn's `load_digits` dataset (8×8 grayscale images with 10 classes) for quick, reproducible development. Fashion-MNIST or a larger dataset can be a future validation milestone after the sparse-inference path is correct.
