
from aqua_nn.autodiff.tensor import Tensor


def test_tensor_initialization():
    x = Tensor(3.0)

    assert x.data == 3.0
    assert x.grad == 0.0
    assert x._prev == set()
    assert callable(x._backward)
    
def test_addition_forward_and_backward():
    x = Tensor(3.0)
    y = Tensor(2.0)

    z = x + y
    z.grad = 1.0
    z._backward()

    assert z.data == 5.0
    assert x.grad == 1.0
    assert y.grad == 1.0


def test_multiplication_forward_and_backward():
    x = Tensor(3.0)
    y = Tensor(2.0)

    z = x * y
    z.grad = 1.0
    z._backward()

    assert z.data == 6.0
    assert x.grad == 2.0
    assert y.grad == 3.0



def test_backward_with_shared_parent():
    x = Tensor(3.0)

    z = x * x + x
    z.backward()

    assert z.data == 12.0
    assert x.grad == 7.0


import numpy as np


def test_shared_parent_gradient():
    x = Tensor(3.0)
    z = x * x + x

    z.backward()

    assert z.data == 12.0
    assert x.grad == 7.0


def test_matrix_multiplication():
    x = Tensor([[1.0, 2.0]])
    w = Tensor([[3.0], [4.0]])

    z = (x @ w).sum()

    assert z.data == 11.0

    z.backward()

    np.testing.assert_array_equal(x.grad, [[3.0, 4.0]])
    np.testing.assert_array_equal(w.grad, [[1.0], [2.0]])


def test_broadcast_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([10.0, 20.0])

    z = (x + b).sum()
    z.backward()

    np.testing.assert_array_equal(b.grad, [2.0, 2.0])


def test_relu_forward_and_backward():
    x = Tensor([-2.0, 0.0, 3.0])
    z = x.relu().sum()

    np.testing.assert_array_equal(z.data, 3.0)

    z.backward()

    np.testing.assert_array_equal(x.grad, [0.0, 0.0, 1.0])
