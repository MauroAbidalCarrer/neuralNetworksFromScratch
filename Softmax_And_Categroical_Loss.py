import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from activation_functions import *
from Loss import *

# This class is a implements a combination of the softmax and cross entropy categorical loss functions derivatives.
# The product of the derivatives of the two functions is 7x faster to run that the two sequantially.
class Softmax_and_Categorical_loss:
    
    def __init__(self):
        self.loss = Categorical_cross_entropy_loss()
        self.activation = SoftMax()

    # Calculate the networks output and the loss.
    def forward(self, last_layer_outputs, categorical_labels):
        self.activation.forward(last_layer_outputs)
        self.loss.calculate_loss(self.activation.outputs, categorical_labels)

    # Calculate gradient with respect to the last layer's outputs.
    def backward(self, outputs_gradients, categorical_labels):
        samples_batch_size = len(categorical_labels)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(categorical_labels.shape) == 2:
            categorical_labels = np.argmax(categorical_labels, axis=1)
        # Copy so we can safely modify
        self.input_gradients = outputs_gradients.copy()
        # Calculate gradient
        self.input_gradients[range(samples_batch_size), categorical_labels] -= 1
        # Normalize gradient
        self.input_gradients = self.input_gradients / samples_batch_size