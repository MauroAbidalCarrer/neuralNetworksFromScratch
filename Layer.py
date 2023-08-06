import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()
from activation_functions import Linear
from Softmax_And_Categroical_Loss import Softmax_and_Categorical_loss

class Layer:

    def __init__(self, input_size, nb_neurons, debug_layer_index, L1_weights_multiplier=0, L1_biases_multiplier=0, L2_weights_multiplier=0, L2_biases_multiplier=0, logs_file=None, activation_function=Linear()):
        # Create a matrix of shape(nb_neurons, nb_inputs) with random values.
        # Since we are using batches of inputs and performing matrix multiplication on them and that
        # because matMul performs the dot product on the rows of the first matrix and the columms of the second instead of row/row,
        # we would need to transpose() the weight matrix for every pass.
        # So instead, we make the matrix of shape (input_size, nb_neurons).
        self.weights = 0.1 * np.random.randn(input_size, nb_neurons)
        # The parameter of the funciton is in parenthesis because it is a tuple of size one.
        # print(self.weights)
        self.biases = np.zeros((1, nb_neurons))
        self.activation_function = activation_function
        self.L1_weights_multiplier = L1_weights_multiplier
        self.L1_biases_multiplier = L1_biases_multiplier
        self.L2_weights_multiplier = L2_weights_multiplier
        self.L2_biases_multiplier = L2_biases_multiplier
        if logs_file and (L1_weights_multiplier or L1_biases_multiplier or L2_weights_multiplier or L2_biases_multiplier):
            debug_str = 'Dense_layer' + str(debug_layer_index) + ':\n'
            debug_str += '\tL1_weights_multiplier: ' + str(self.L1_weights_multiplier) + '\n'
            debug_str += '\tL1_biases_multiplier: ' + str(self.L1_biases_multiplier) + '\n'
            debug_str += '\tL2_weights_multiplier: ' + str(self.L2_weights_multiplier) + '\n'
            debug_str += '\tL2_biases_multiplier: ' + str(self.L2_biases_multiplier) + '\n'
            logs_file.write(debug_str)
            print(debug_str)

    def forward(self, inputs):
        self.inputs = inputs
        linear_outputs = np.dot(inputs, self.weights) + self.biases   # Perform dot product of the weights on each inputs(Should maybe use @ ?)
                                                                    # No need to transpose the weight matrix since its shape is (input_size, nb_neurons).
        self.outputs = self.activation_function.forward(linear_outputs)
        return self.outputs

    # Calculates the gradient of parameters from the output_gradients
    # Also calculates the gradients of the inputs which will be used by the previous layer to calculate its parameters gradients.
    def backward(self, output_gradients):
        if not isinstance(self.activation_function, Softmax_and_Categorical_loss):
            output_gradients = self.activation_function.backward(output_gradients)
        # Gradients on parameters
        self.weights_gradient = np.dot(self.inputs.T, output_gradients)
        self.biases_gradient = np.sum(output_gradients, axis=0, keepdims=True)
        # Gradient on values
        self.inputs_gradients = np.dot(output_gradients, self.weights.T)

        # L1 regularization
        L1_regularization_weights_matrix = np.ones_like(self.weights)
        L1_regularization_weights_matrix[self.weights < 0] = -1
        self.weights_gradient += L1_regularization_weights_matrix * self.L1_weights_multiplier
        L1_regularization_biases_vector = np.ones_like(self.biases)
        L1_regularization_biases_vector[self.biases < 0] = -1
        self.weights_gradient += L1_regularization_biases_vector * self.L1_biases_multiplier
        # L2 regularization
        self.weights_gradient += 2 * self.weights * self.L2_weights_multiplier
        self.biases_gradient += 2 * self.biases * self.L2_weights_multiplier
        return self.inputs_gradients