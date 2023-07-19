import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()

class SGD_Optimizer:

    def __init__(self, learning_rate=1, decay_rate=0.):
        self.starting_learning_rate = learning_rate
        self.learning_rate = self.starting_learning_rate
        self.decay_rate = decay_rate
        self.nb_update_iterations = 0

    def pre_update_layers_params(self):
        self.nb_update_iterations += 1
        if (self.decay_rate):
            self.learning_rate = self.starting_learning_rate / (self.decay_rate * self.nb_update_iterations)
            if (self.nb_update_iterations % 1000):
                print('self.nb_update_iterations: ', self.nb_update_iterations, ' ,', self.learning_rate)

    def update_layer_params(self, layer): 
        layer.weights -= self.learning_rate * layer.weights_gradient
        layer.biases -= self.learning_rate * layer.biases_gradient