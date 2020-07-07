import numpy as np
from nn_tutorials.datasets import spiral_data

np.random.seed(0)

X, y = spiral_data(100, 3)
print(y.shape)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)  # as per tutorials value is np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.outptut = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()
layer2 = LayerDense(5, 2)
activation2 = ActivationReLU()

layer1.forward(X)
# print(layer1.outptut)
activation1.forward(layer1.outptut)
print(activation1.output)

layer2.forward(layer1.outptut)
# print(layer2.outptut)
activation2.forward(layer2.outptut)
print(activation2.output)
