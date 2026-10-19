
import numpy as np


class Adam:
    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        """Reset parameter gradients before the next training batch."""
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)

    def step(self):
        """Update parameters while keeping masked weights permanently zero."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            mask = getattr(param, "_mask", None)

            if mask is not None:
                mask = np.asarray(mask, dtype=np.float64)

                if mask.shape != param.data.shape:
                    raise ValueError("Parameter mask shape mismatch")

                # Ignore gradients for masked connections.
                grad = param.grad * mask
            else:
                grad = param.grad

            # Adam first and second moments.
            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1) * grad
            )
            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2) * (grad ** 2)
            )

            if mask is not None:
                # Remove stale momentum from masked connections.
                self.m[i] *= mask
                self.v[i] *= mask

            # Bias correction.
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Adam parameter update.
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if mask is not None:
                # A masked weight must remain exactly zero.
                param.data *= mask
