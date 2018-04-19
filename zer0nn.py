import numpy as np
from zer0nn_layers import *

class NeuralNetwork(object):
    def __init__(self, dims, layers, learning_rate, learning_decay):
        self.layers = [None, ]
        for i, dim in enumerate(dims[1:]):
            self.layers.append(SpaceLayerFactory.layer(layers[i], dims[i], dim, i + 1))

        self.lr = learning_rate

        self.decay = learning_decay
        self.counter = 0

    def learning_rate(self):
        self.counter += 1
        if 0 == self.counter % 100:
            self.lr *= self.decay
        return self.lr

    def __forward_pass__(self, inputs, name = "input"):
        self.layers[0] = Input(name, inputs)
        for layer in self.layers[1:]:
            inputs = layer.fire_signal(inputs)
        return inputs

    def train(self, features, targets):
        ## Forward Pass ##
        inputs = self.__forward_pass__(features, "train_input")

        ## Backward Pass ##
        error = self.layers[-1].calc_err(inputs, targets)
        for i, layer in enumerate(reversed(self.layers[1:])):
            error = layer.eval_signal(error, self.layers[-1-i-1].signal_out())

        ## Update network weights ##
        for layer in self.layers[1:]:
            layer.update(self.learning_rate())

    def run(self, features):
        return self.__forward_pass__(features, "test_input")
