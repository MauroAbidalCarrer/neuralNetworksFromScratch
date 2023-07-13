import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()

class SGD_Optimizer:

    def __init__(self, Learning_rate=1):
        self.Learning_rate = Learning_rate

    def update_layer_params(self, layer): 
        # print('Optimization:')
        # print('layer.weights.shape: ', layer.weights.shape, ', layer.weights_gradient.shape: ', layer.weights_gradient.shape)
        layer.weights -= self.Learning_rate * layer.weights_gradient
        # biasses_offset = layer.biases_gradient
        # print('layer.biases_gradient.shape: ', layer.biases_gradient.shape)
        # print('layer.biases.shape: ', layer.biases.shape)
        # layer.biases -= biasses_offset
        layer.biases -= self.Learning_rate * layer.biases_gradient