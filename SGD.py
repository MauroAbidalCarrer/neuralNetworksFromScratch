import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()

class SGD_Optimizer:

    def __init__(self, learning_rate=1, decay_rate=0., momentum=0., logs_file=None):
        self.initial_learning_rate = learning_rate
        self.learning_rate = self.initial_learning_rate
        self.decay_rate = decay_rate
        self.nb_update_iterations = 0
        self.momentum = momentum
        # Debugging
        debug_str = 'Optimizer hyper parameters:\n'
        debug_str += 'learning rate:\t\t' + str(learning_rate) + '\n'
        debug_str += 'learning_decay_rate:\t' + str(decay_rate) + '\n'
        debug_str += 'momentum:\t\t' + str(momentum) + '\n\n'
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
        if self.momentum:
            # Assert that the layer has the weights_momentum, and by extension, the biases momemtum attributes with hasattr.
            if not hasattr(layer, 'weights_gradient_momentum'):
                layer.weights_gradient_momentum = np.zeros_like(layer.weights_gradient)
                layer.biases_gradient_momemtum = np.zeros_like(layer.biases_gradient)
            # Update the layer parameters with the current and previous gradient.
            layer.weights -= self.learning_rate * layer.weights_gradient - layer.weights_gradient_momentum * self.momentum
            layer.biases -= self.learning_rate * layer.biases_gradient -  layer.biases_gradient_momemtum * self.momentum
            # Update the gradient momentum to be the current gradient.
            layer.weights_gradient_momentum = layer.weights_gradient
            layer.biases_gradient_momemtum = layer.biases_gradient
        else:
            layer.weights -= self.learning_rate * layer.weights_gradient
            layer.biases -= self.learning_rate * layer.biases_gradient

    def post_update_layer_params(self):
        self.nb_update_iterations += 1
