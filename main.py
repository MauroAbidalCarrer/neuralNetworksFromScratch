import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from nnfs.datasets import sine_data
from activation_functions import *
from Layer import *
from plot import *
from Softmax_And_Categroical_Loss import *
from SGD import *
from AdaptiveGradient import *
from Root_Mean_Square_Propagation import *
from AdaptiveMomentum import *


logs_file = open('logs.txt', '+a')

# Create dataset represented as a tuple of 2D sample vectors and categorical labels targets.
nb_classes = 2
nb_training_samples = 100
training_samples, expected_training_values = sine_data()
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
# one_hot_targets = np.eye(nb_classes)[training_categorical_labels]


# Define the layers of the network and the function that will calculate the loss.
layer1 = Layer(1, 64, 0)
activation1 = Relu()
layer2 = Layer(64, 64, 1)
activation2 = Relu()
layer3 = Layer(64, 1, 2)
loss_function = SquaredMean_Loss()


# Training
optimizer = Adam_Optimizer(learning_rate=0.005, decay_rate=1e-3, logs_file=logs_file)
nb_epochs = 10001

def forward_pass(samples):
    # forward pass
    layer1.forward(samples)
    activation1.forward(layer1.outputs)
    layer2.forward(activation1.outputs)
    activation2.forward(layer2.outputs)
    layer3.forward(activation2.outputs)
    # activation2.forward(layer2.outputs)


for epoch in range(nb_epochs): 
    forward_pass(training_samples)

    # backward pass: calculate gradients for the parameters and the inputs of each layer/activation function
    loss_function.backward(layer3.outputs, expected_training_values)
    layer3.backward(loss_function.inputs_gadients)
    activation2.backward(layer3.inputs_gradients)
    layer2.backward(activation2.inputs_gradients)
    activation1.backward(layer2.inputs_gradients)
    layer1.backward(activation1.inputs_gradients)

    # Optimization: apply negative of gradients 
    optimizer.pre_update_layers_params()

    optimizer.update_layer_params(layer1)
    optimizer.update_layer_params(layer2)
    optimizer.update_layer_params(layer3)

    optimizer.post_update_layer_params()

# Debugging

def calculate_mean_accuracy(expected_outputs):
    # accuracy_margin defines the margin of error allowed for the nn_outputs.
    # It is relative to the standard deviation of the expected_outputs (np.std(expected_outputs)).
    # https://en.wikipedia.org/wiki/Standard_deviation
    # Standard deviation is a measure of how much variation there is in the expected outputs batch.
    # This is because we want the network to output data that is similar from the training data.
    accuracy_margin = np.std(expected_outputs) / 250
    return np.mean(np.abs(expected_outputs - layer3.outputs) <= accuracy_margin)


debug_str = 'nb training samples' + str(nb_training_samples) + '\n'
debug_str += 'Training loss: ' + str(loss_function.calculate_mean_loss(activation2.outputs, expected_training_values)) + '\n'
debug_str += 'Training accuracy: ' + str(calculate_mean_accuracy(expected_training_values)) + '\n'
# Measure loss and accuracy on test dataset
nb_test_samples = 100
test_samples, expected_test_values = sine_data()
# test_bianary_labels = test_categorical_labels.reshape(-1, 1)
forward_pass(test_samples)
debug_str += 'nb test samples: ' + str(nb_test_samples) + '\n'
debug_str += 'Test loss: ' + str(loss_function.calculate_mean_loss(activation2.outputs, expected_test_values)) + '\n'
debug_str += 'Test accuracy: ' + str(calculate_mean_accuracy(expected_test_values)) + '\n'
# Write debug string to CLI and logs file
print(debug_str, end="")
logs_file.write(debug_str)
logs_file.write('=======================================\n')
logs_file.close()

#Daw decision boundry
# Generate a grid of points (replace the ranges as necessary) for background(bg) samples
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
bg_samples = np.array(np.c_[X.ravel(), Y.ravel()])


import matplotlib.pyplot as plt

X_test, y_test = sine_data()

layer1.forward(X_test)
activation1.forward(layer1.outputs)
layer2.forward(activation1.outputs)
activation2.forward(layer2.outputs)
layer3.forward(activation2.outputs)

plt.plot(X_test, y_test)
plt.plot(X_test, layer3.outputs)
plt.show()