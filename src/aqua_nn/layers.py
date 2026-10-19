
import numpy as np
from aqua_nn.autodiff.tensor import Tensor


class Dense:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_features)

        self.weight = Tensor(
            rng.standard_normal((in_features, out_features)) * scale
        )
        self.bias = Tensor(np.zeros(out_features))

        # Only weights are prunable; biases remain untouched.
        self.mask = np.ones_like(self.weight.data)
        self.weight._mask = self.mask

    def __call__(self, x):
        effective_weight = self.weight * Tensor(self.mask)
        return x @ effective_weight + self.bias

    def set_mask(self, new_mask):
        """Apply a binary mask and permanently zero newly masked weights."""
        new_mask = np.asarray(new_mask, dtype=np.float64)

        if new_mask.shape != self.weight.data.shape:
            raise ValueError("Mask shape must match weight shape")
        if not np.all((new_mask == 0) | (new_mask == 1)):
            raise ValueError("Mask must contain only 0 and 1")

        self.mask = new_mask.copy()
        self.weight._mask = self.mask
        self.weight.data *= self.mask

    def parameters(self):
        return [self.weight, self.bias]


def relu(x):
    return x.relu()
