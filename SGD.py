import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()

class SGD_Optimizer:

    def __init__(self, learning_rate=1, decay_rate=0.):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.nb_update_iterations = 0

    def pre_update_layer_params(self):
        if (self.decay_rate):
            self.learning_rate = 1 / (self.decay_rate * self.nb_update_iterations)

    def update_layer_params(self, layer): 
        layer.weights -= self.learning_rate * layer.weights_gradient
        layer.biases -= self.learning_rate * layer.biases_gradient

    def post_update_layer_params(self):
        self.nb_update_iterations += 1
