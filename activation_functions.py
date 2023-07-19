import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
import sys
import warnings

class Relu:

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, output_gradients):
        self.inputs_gradients = output_gradients.copy()
        self.inputs_gradients[self.inputs <= 0] = 0



class SoftMax:
    def forward(self, inputs):
        # max_log = np.log(sys.float_info.max) - 1
        # print("max log: ", max_log)
        # print('exp(max_log): ', np.exp(max_log))
        # print('min float: ', -sys.float_info.max)
        # exponantiate the input.
        # Clip inputs to log of float max to prevent overflow when exponentiating the inputs
        # inputs = np.clip(inputs, -sys.float_info.max, np.log(sys.float_info.max))
        exp_inputs = np.exp(inputs)
        # Devide it by its sum.
        # Since we are using a batch if imputs (i.e a matrix) we use the following parameters for sum: 
        # - "axis=1" specifies that we want to add only the components of the vectors.
        # - "keepdims=True" specifies that we want the result to keep the same number of dimensions(but not the same shape).
        # This way we get a matrix as output of the sum instead of a vector.
        # Having a matrix of shape (nb_inputs, 1) allows us to use it as denomintor for exp_inputs which is also a matrix.
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs