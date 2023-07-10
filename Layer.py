import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()
from nnfs.datasets import spiral_data

class Layer:

    def __init__(self, input_size, nb_neurons):
        # Create a matrix of shape(nb_neurons, nb_inputs) with random values.
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first matrix and the columms of the second instead of row/row,
        # we would need to transpose() the weight matrix for every pass.
        # So instead, we make the matrix of shape (input_size, nb_neurons).
        self.weights = np.random.randn(input_size, nb_neurons)
        # The parameter of the funciton is in parenthesis because it is a tuple of size one.
        # print(self.weights)
        self.biases = np.zeros((nb_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights)                        # Perform dot product of the weights on each inputs(Should maybe use @ ?)
                                                                           # No need to transpose the weight matrix since its shape is (input_size, nb_neurons).
        self.outputs = self.outputs + self.biases                          # Add the biases on each row/input.