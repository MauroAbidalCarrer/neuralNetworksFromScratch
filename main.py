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
samples, targets = spiral_data(samples=100, classes=3)

# plot_samples(samples, targets)

layer1 = Layer(2, 3, Relu)
# print(X.__len__())
layer1.forward_pass(samples)
# print(layer1.output)
layer2 = Layer(3, 3, SoftMax)
layer2.forward_pass(layer1.output)
# print(layer2.output[:5])

loss_function = Categorical_cross_entropy_loss()
# print(targets)
average_loss = loss_function.calculate_average_loss(layer2.output, targets)
print('average_loss: ', average_loss)