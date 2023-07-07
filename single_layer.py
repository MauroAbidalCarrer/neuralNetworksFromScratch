import sys
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from activation_functions import *
from plot import *


class Layer:

    def __init__(self, input_size, nb_neurons, activation_function):
        self.nb_inputs = input_size
        self.nb_neurons = nb_neurons
        # Create a matrix of shape(nb_neurons, nb_inputs) with random values.
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first matrix and the columms of the second instead of row/row,
        # we would need to transpose() the weight matrix for every pass.
        # So instead, we make the matrix of shape (input_size, nb_neurons).
        self.weights = np.random.randn(input_size, nb_neurons)
        # The parameter of the funciton is in parenthesis because it is a tuple of size one.
        # print(self.weights)
        self.biases = np.zeros((nb_neurons))
        self.activation_function = activation_function

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights)                        # Perform dot product of the weights on each inputs(Should maybe use @ ?)
                                                                            # No need to transpose the weight matrix since its shape is (input_size, nb_neurons).
        self.output = self.output + self.biases                             # Add the biases on each row/input.
        self.output = self.activation_function.forward(self.output)         # Apply the activation function.



#main

# Create dataset
samples, targets = spiral_data(samples=100, classes=3)

# plot_samples(samples, targets)

layer1 = Layer(2, 3, Relu)
# print(X.__len__())
layer1.forward_pass(samples)
# print(layer1.output)
layer2 = Layer(3, 3, SoftMax)
layer2.forward_pass(layer1.output)
print(layer2.output)