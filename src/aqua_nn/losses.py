
import numpy as np

from aqua_nn.autodiff.tensor import Tensor


def softmax_cross_entropy(logits, targets):
    """Mean cross-entropy for logits shaped (batch, classes)."""
    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")

    targets = np.asarray(targets, dtype=np.int64)

    batch_size, num_classes = logits.data.shape

    if targets.shape != (batch_size,):
        raise ValueError("targets must have shape (batch,)")
    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("target class index out of range")

    # Stable softmax: subtract each row's maximum.
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(
        axis=1, keepdims=True
    )

    log_probs = (
        shifted
        - np.log(exp_scores.sum(axis=1, keepdims=True))
    )
    loss_value = -log_probs[np.arange(batch_size), targets].mean()

    loss = Tensor(loss_value)
    loss._prev = {logits}

    def _backward():
        grad = probabilities.copy()
        grad[np.arange(batch_size), targets] -= 1.0
        logits.grad += grad * (loss.grad / batch_size)

    loss._backward = _backward
    return loss


def accuracy(logits, targets):
    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == np.asarray(targets)))
