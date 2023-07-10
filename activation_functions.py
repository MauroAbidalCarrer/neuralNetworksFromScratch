import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()


class Relu:

    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        return self.outputs

class SoftMax:

    def forward(self, inputs):
        # exponantiate the input.
        exp_inputs = np.exp(inputs)                                     
        # Devide it by its sum.
        # Since we are using a batch if imputs (i.e a matrix) we use the following parameters for sum: 
        # - "axis=1" specifies that we want to add only the components of the vectors.
        # - "keepdims=True" specifies that we want the result to keep the same number of dimensions(but not the same shape).
        # This way we get a matrix as output of the sum instead of a vector.
        # Having a matrix of shape (nb_inputs, 1) allows us to use it as denomintor for exp_inputs which is also a matrix.
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs
    
    # def calculate_gradient(self, prev_grad):
