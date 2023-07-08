import numpy as np
import nnfs
from nnfs.datasets import spiral_data
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from activation_functions import *
from Layer import *
from plot import *
from Loss import *

# Create dataset represented as a tuple of 2D sample vectors and categorical labels targets.
nb_classes = 3
samples, categorical_targets = spiral_data(samples=100, classes=nb_classes)
# 'np.eye(num_classes)' generates a square matrix (2D array) with the 
# number of rows and columns equal to 'num_classes'. In this matrix, 
# the diagonal elements are 1's, and all other elements are 0's. This 
# is also known as an "identity matrix". For example, if num_classes is 3, 
# np.eye(num_classes) would be:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
# Each row in this matrix can be seen as a "template" for a one-hot 
# encoded label for each class.

# 'np.eye(num_classes)[labels]' uses the 'labels' array to select rows 
# from the identity matrix. Since the 'labels' array contains the class 
# labels (0, 1, 2), it will select the 0th, 1st, or 2nd row from the 
# identity matrix, depending on the label. This operation turns the 
# 'labels' array into a one-hot encoded 2D array.
one_hot_targets = np.eye(nb_classes)[categorical_targets]


# Define the layers of the network and the function that will calculate the loss.
layer1 = Layer(2, 3, Relu)
layer2 = Layer(3, 3, SoftMax)
loss_function = Categorical_cross_entropy_loss()

# Get the output of the network and calculate the average loss.
layer1.forward_pass(samples)
layer2.forward_pass(layer1.output)
average_loss = loss_function.calculate_average_loss(layer2.output, categorical_targets)

print('average_loss: ', average_loss)