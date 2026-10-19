from aqua_nn.autodiff.tensor import Tensor

x1 = Tensor(2.0)
x2 = Tensor(3.0)

w1 = Tensor(0.5)
w2 = Tensor(0.2)

b = Tensor(1.0)

z = x1 * w1 + x2 * w2 + b
a = max(0, z.data)
print("After ReLU:", a)

print("Neuron output:", z.data)