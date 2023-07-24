import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()
from Constants import *

class AdaGrad_Optimizer:

    def __init__(self, learning_rate=1, decay_rate=0., logs_file=None):
        self.initial_learning_rate = learning_rate
        self.learning_rate = self.initial_learning_rate
        self.decay_rate = decay_rate
        self.nb_update_iterations = 0
        # Debugging
        debug_str = 'Optimizer hyper parameters:\n'
        debug_str += 'learning rate:\t\t' + str(learning_rate) + '\n'
        debug_str += 'learning_decay_rate:\t' + str(decay_rate) + '\n'
        print(debug_str, end="")
        if logs_file:
            logs_file.write(debug_str)

    def pre_update_layers_params(self):
        self.nb_update_iterations += 1
        if (self.decay_rate):
            # Don't replace the '1 +' by setting 'self.nb_update_iterations = 1' because it will get multiplied by decay rate.
            # We want the denominator to always be at least be 1 otherwise increase the value self.learning_rate instead of decreasing it.
            self.learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.nb_update_iterations)

    def update_layer_params(self, layer):
        # Add weights and biases updates sum cache if the layer doesn't already have it .
        if not hasattr(layer, 'weights_gradient_sum_cache'):
            layer.weights_gradient_sum_cache = np.zeros_like(layer.weights_gradient)
            layer.biases_gradient_sum_cache = np.zeros_like(layer.biases_gradient)
        # Update gradient sum cache. 
        # We update the squared root of the gradient and to get the absolute value of the gradietn AND to make the cache grow slower(when getting the squared root) than if we were to simply add the absolute values.
        layer.weights_gradient_sum_cache += layer.weights_gradient ** 2 
        layer.biases_gradient_sum_cache += layer.biases_gradient ** 2
        # Update layer parameters.
        layer.weights -= self.learning_rate * layer.weights_gradient / (np.sqrt(layer.weights_gradient_sum_cache))
        layer.biases -= self.learning_rate * layer.biases_gradient / (np.sqrt(layer.biases_gradient_sum_cache))

    def post_update_layer_params(self):
        self.nb_update_iterations += 1
