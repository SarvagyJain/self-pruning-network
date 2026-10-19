

import numpy as np

from aqua_nn.autodiff.tensor import Tensor
from aqua_nn.models import MLP


def test_mlp_output_shape_and_gradients():
    model = MLP()
    x = Tensor(np.ones((4, 64)))

    logits = model(x)

    assert logits.data.shape == (4, 10)

    logits.sum().backward()

    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.data.shape
        assert np.isfinite(parameter.grad).all()
