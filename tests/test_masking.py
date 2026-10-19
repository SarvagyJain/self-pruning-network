
import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.layers import Dense


def test_masked_weights_do_not_affect_forward():
    layer = Dense(2, 1, seed=1)
    layer.weight.data[:] = [[2.0], [3.0]]
    layer.set_mask([[1.0], [0.0]])

    output = layer(Tensor([[4.0, 5.0]]))

    np.testing.assert_allclose(output.data, [[8.0]])


def test_masked_weight_has_zero_gradient():
    layer = Dense(2, 1, seed=1)
    layer.set_mask([[1.0], [0.0]])

    layer(Tensor([[4.0, 5.0]])).sum().backward()

    assert layer.weight.grad[1, 0] == 0.0
    assert layer.weight.grad[0, 0] != 0.0


def test_mask_does_not_change_bias():
    layer = Dense(2, 1, seed=1)
    layer.bias.data[:] = [7.0]
    layer.set_mask([[0.0], [0.0]])

    output = layer(Tensor([[4.0, 5.0]]))

    np.testing.assert_allclose(output.data, [[7.0]])
    np.testing.assert_allclose(layer.bias.data, [7.0])


def test_masked_weights_are_exactly_zero():
    layer = Dense(2, 1, seed=1)
    layer.weight.data[:] = [[2.0], [3.0]]
    layer.set_mask([[1.0], [0.0]])

    assert layer.weight.data[1, 0] == 0.0
