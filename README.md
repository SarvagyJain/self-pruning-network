# The Self-Pruning Network

A neural-network library built from scratch using Python and NumPy. The project will implement reverse-mode automatic differentiation, a trainable multilayer perceptron, mask-aware optimization, gradual self-pruning, and sparse-aware inference.

## Engineering principles

- Implement the core tensor engine, neural-network layers, optimizer, training loop, and pruning logic ourselves.
- Use NumPy for numerical operations.
- Use scikit-learn only for dataset loading, not for model training, autodiff, optimization, or pruning.
- Validate gradients and masking behavior with automated tests.
- Report measured accuracy, sparsity, operation counts, and inference latency honestly.

## Development status

- [x] Project environment and dependency management
- [x] Deterministic smoke-test dataset loader
- [ ] Reverse-mode autodiff engine
- [ ] Dense MLP and Adam optimizer
- [ ] Self-pruning and sparse-aware inference
- [ ] Reproducible experiments and benchmarks

## Setup

```powershell
uv sync
uv run pytest
```

The `load_digits` dataset is used for fast smoke tests. Fashion-MNIST is planned for the main experiments.
