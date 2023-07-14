import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from nnfs.datasets import spiral_data
from activation_functions import *
from Layer import *
from plot import *
from Softmax_And_Categroical_Loss import *
from SGD import *

# Create dataset represented as a tuple of 2D sample vectors and categorical labels targets.
nb_classes = 3
samples, categorical_labels = spiral_data(samples=100, classes=nb_classes)
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
one_hot_targets = np.eye(nb_classes)[categorical_labels]


# Define the layers of the network and the function that will calculate the loss.
layer1 = Layer(2, 64)
activation1 = Relu()
layer2 = Layer(64, 3)
last_activation_and_loss = Softmax_and_Categorical_loss()
optimizer = SGD_Optimizer(Learning_rate=0.1)

nb_epochs = 100001

for epoch in range(nb_epochs): 

    # forward pass
    layer1.forward(samples)
    activation1.forward(layer1.outputs)
    layer2.forward(activation1.outputs)
    last_activation_and_loss.forward(layer2.outputs, categorical_labels)

    # backward pass
    last_activation_and_loss.backward(last_activation_and_loss.activation.outputs, categorical_labels)
    layer2.backward(last_activation_and_loss.input_gradients)
    activation1.backward(layer2.inputs_gradients)
    layer1.backward(activation1.inputs_gradients)

    # Optimization
    optimizer.update_layer_params(layer1)
    optimizer.update_layer_params(layer2)

    # if not epoch % 1000:
print('Loss: ', np.mean(last_activation_and_loss.loss.losses))