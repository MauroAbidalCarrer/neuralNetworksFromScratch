import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()

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
        self.biases = np.zeros((1, nb_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases   # Perform dot product of the weights on each inputs(Should maybe use @ ?)
                                                                    # No need to transpose the weight matrix since its shape is (input_size, nb_neurons).

    # Calculates the gradient of parameters from the output_gradients
    # Also calculates the gradients of the inputs which will be used by the previous layer to calculate its parameters gradients.
    def backward(self, output_gradients):
        # print('dense layer output_gradients shape: ', output_gradients.shape)
        # Gradients on parameters
        self.weights_gradient = np.dot(self.inputs.T, output_gradients)
        # print('self.inputs.T[:, :3]:\n', self.inputs.T[:, :3])
        # print('output_gradients[:3]:\n', output_gradients[:3])
        self.biases_gradient = np.sum(output_gradients, axis=0, keepdims=True)
        # print('self.biases_gradient.shape: ', self.biases_gradient.shape)
        # Gradient on values
        self.inputs_gradients = np.dot(output_gradients, self.weights.T)