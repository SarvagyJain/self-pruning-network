
import numpy as np
from aqua_nn.autodiff.tensor import Tensor


def test_subtraction_and_division_gradients():
    x = Tensor(6.0)
    y = Tensor(2.0)
    ((x - y) + (x / y)).backward()

    assert np.isclose(x.grad, 1.5)
    assert np.isclose(y.grad, -2.5)


def test_broadcast_division_gradient_shapes():
    x = Tensor(np.array([[2.0, 4.0], [6.0, 8.0]]))
    y = Tensor(np.array([2.0, 4.0]))

    (x / y).sum().backward()

    np.testing.assert_allclose(x.grad, [[0.5, 0.25], [0.5, 0.25]])
    np.testing.assert_allclose(y.grad, [-2.0, -0.75])


def test_mean_gradient():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    x.mean().backward()

    np.testing.assert_allclose(x.grad, np.full((2, 2), 0.25))


def test_tanh_gradient():
    x = Tensor(np.array([0.2, -0.4]))
    x.tanh().sum().backward()

    expected = 1.0 - np.tanh([0.2, -0.4]) ** 2
    np.testing.assert_allclose(x.grad, expected)


def test_sigmoid_gradient_and_extreme_values():
    x = Tensor(np.array([-1000.0, 0.0, 1000.0]))
    y = x.sigmoid()
    y.sum().backward()

    assert np.all(np.isfinite(y.data))
    np.testing.assert_allclose(y.data, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(x.grad, y.data * (1.0 - y.data))


def test_intermediate_gradients_reset_between_backward_calls():
    x = Tensor(3.0)
    y = x * x

    y.backward()
    np.testing.assert_allclose(x.grad, 6.0)

    # Leaf gradients accumulate by design; intermediate gradients refresh.
    y.backward()
    np.testing.assert_allclose(x.grad, 12.0)
