import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.layers import Dense, relu


class MLP:
    def __init__(self, input_size=64, hidden1=128, hidden2=64, num_classes=10, seed=42):
        self.layer1 = Dense(input_size, hidden1, seed=seed)
        self.layer2 = Dense(hidden1, hidden2, seed=seed + 1)
        self.layer3 = Dense(hidden2, num_classes, seed=seed + 2)

    def __call__(self, x):
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        return self.layer3(x)

    def parameters(self):
        return (
            self.layer1.parameters()
            + self.layer2.parameters()
            + self.layer3.parameters()
        )

    def _inference_layers(self):
        return (
            (self.layer1, True),
            (self.layer2, True),
            (self.layer3, False),
        )

    def predict_logits_dense(self, x):
        """Run ordinary dense NumPy inference while respecting pruning masks."""
        activations = np.asarray(x, dtype=np.float64)
        if activations.ndim != 2 or activations.shape[1] != self.layer1.weight.data.shape[0]:
            raise ValueError(
                f"x must have shape (batch, {self.layer1.weight.data.shape[0]})"
            )

        for layer, apply_relu in self._inference_layers():
            weights = layer.weight.data * layer.mask
            activations = activations @ weights + layer.bias.data
            if apply_relu:
                np.maximum(activations, 0.0, out=activations)
        return activations

    def to_sparse_weights(self):
        """Return a CSR snapshot of the masked weight matrices.

        The snapshot is separate from the model parameters. Rebuild it after
        training or pruning changes the weights or masks.
        """
        try:
            from scipy.sparse import csr_matrix
        except ImportError as exc:
            raise ImportError(
                "CSR inference requires SciPy. Install dependencies using uv sync."
            ) from exc

        return [
            csr_matrix(layer.weight.data * layer.mask)
            for layer, _ in self._inference_layers()
        ]

    def predict_logits_sparse(self, x, sparse_weights=None):
        """Run inference using CSR weights and dense batch activations."""
        try:
            from scipy.sparse import issparse
        except ImportError as exc:
            raise ImportError(
                "CSR inference requires SciPy. Install dependencies using uv sync."
            ) from exc

        activations = np.asarray(x, dtype=np.float64)
        if activations.ndim != 2 or activations.shape[1] != self.layer1.weight.data.shape[0]:
            raise ValueError(
                f"x must have shape (batch, {self.layer1.weight.data.shape[0]})"
            )

        if sparse_weights is None:
            sparse_weights = self.to_sparse_weights()
        if len(sparse_weights) != 3 or not all(issparse(w) for w in sparse_weights):
            raise ValueError("sparse_weights must contain three SciPy sparse matrices")

        for (layer, apply_relu), weights in zip(self._inference_layers(), sparse_weights):
            if weights.shape != layer.weight.data.shape:
                raise ValueError("Sparse weight shape does not match model layer")

            # W has shape (input_features, output_features). Computing W.T @ X.T
            # uses CSR sparse-dense multiplication and returns dense activations.
            activations = np.asarray((weights.T @ activations.T).T, dtype=np.float64)
            activations += layer.bias.data
            if apply_relu:
                np.maximum(activations, 0.0, out=activations)
        return activations
