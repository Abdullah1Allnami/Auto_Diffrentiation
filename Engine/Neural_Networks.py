import random
from typing import List
from ..engine.engine import Scalar



class Module:
    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, num_inputs: int, nonlinear: bool = True):
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Scalar(0)
        self.nonlinear = nonlinear

    def __call__(self, inputs):
        activation = sum((input_val * weight for input_val, weight in zip(inputs, self.weights))) + self.bias
        return activation.relu() if self.nonlinear else activation

    def parameters(self):
        return self.weights + [self.bias]

    def __repr__(self) -> str:
        return f'{"ReLU" if self.nonlinear else "Linear"} Neuron({len(self.weights)}, bias={self.bias})'


class Layer(Module):
    def __init__(self, num_inputs: int, num_outputs: int, nonlinear: bool):
        self.neurons = [Neuron(num_inputs, nonlinear) for _ in range(num_outputs)]

    def __call__(self, inputs):
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"


class MLP(Module):
    def __init__(self, num_inputs: int, num_outputs: List[int]):
        layer_sizes = [num_inputs] + num_outputs
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1], i != len(num_outputs) - 1)
                       for i in range(len(num_outputs))]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
