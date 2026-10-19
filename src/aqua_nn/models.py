
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
