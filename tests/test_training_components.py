

import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.layers import Dense, relu
from aqua_nn.losses import softmax_cross_entropy, accuracy
from aqua_nn.optim import Adam


def test_dense_layer_forward_and_backward():
    layer = Dense(4, 3, seed=42)
    x = Tensor(np.ones((2, 4)))

    output = layer(x)
    assert output.data.shape == (2, 3)

    output.sum().backward()

    assert layer.weight.grad.shape == (4, 3)
    assert layer.bias.grad.shape == (3,)
    assert np.isfinite(layer.weight.grad).all()


def test_relu_layer():
    x = Tensor(np.array([[-1.0, 2.0]]))
    output = relu(x)

    np.testing.assert_array_equal(output.data, [[0.0, 2.0]])

    output.sum().backward()
    np.testing.assert_array_equal(x.grad, [[0.0, 1.0]])


def test_cross_entropy_and_adam_update():
    logits = Tensor(
        np.array([[2.0, 0.0], [0.0, 2.0]])
    )
    targets = np.array([0, 1])

    loss = softmax_cross_entropy(logits, targets)
    assert loss.data.item() > 0
    assert accuracy(logits, targets) == 1.0

    loss.backward()
    assert np.isfinite(logits.grad).all()

    parameter = Tensor(np.array([1.0, -1.0]))
    parameter.grad[:] = [0.5, -0.5]
    before = parameter.data.copy()

    optimizer = Adam([parameter], lr=0.01)
    optimizer.step()

    assert not np.array_equal(before, parameter.data)
    assert optimizer.step_count == 1
