
import numpy as np


def _sum_to_shape(grad, shape):
    """Reduce a broadcast gradient to the original operand shape."""
    grad = np.asarray(grad, dtype=np.float64)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set()
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _sum_to_shape(out.grad, self.data.shape)
            other.grad += _sum_to_shape(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data)
        out._prev = {self}

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _sum_to_shape(
                other.data * out.grad, self.data.shape
            )
            other.grad += _sum_to_shape(
                self.data * out.grad, other.data.shape
            )

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _sum_to_shape(
                out.grad / other.data, self.data.shape
            )
            other.grad += _sum_to_shape(
                -out.grad * self.data / (other.data ** 2),
                other.data.shape,
            )

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            other.grad += np.swapaxes(self.data, -1, -2) @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in sorted(a % self.data.ndim for a in axes):
                    grad = np.expand_dims(grad, axis=ax)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.data.shape[a] for a in axes]))

        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data))
        out._prev = {self}

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        values = np.tanh(self.data)
        out = Tensor(values)
        out._prev = {self}

        def _backward():
            self.grad += (1.0 - values**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        # Stable sigmoid: avoids overflow for large positive/negative values.
        values = np.empty_like(self.data)
        positive = self.data >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        values[~positive] = exp_x / (1.0 + exp_x)

        out = Tensor(values)
        out._prev = {self}

        def _backward():
            self.grad += values * (1.0 - values) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output. "
                "Use sum() or mean() to reduce an array first."
            )

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor in visited:
                return
            visited.add(tensor)
            for parent in tensor._prev:
                build_topo(parent)
            topo.append(tensor)

        build_topo(self)

        # Refresh intermediate gradients but preserve leaf/parameter gradients.
        for tensor in topo:
            if tensor._prev:
                tensor.grad = np.zeros_like(tensor.data, dtype=np.float64)

        self.grad = np.ones_like(self.data, dtype=np.float64)

        for tensor in reversed(topo):
            tensor._backward()
