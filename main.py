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
from AdaptiveGradient import *
from Root_Mean_Square_Propagation import *
from AdaptiveMomentum import *


logs_file = open('logs.txt', '+a')

# Create dataset represented as a tuple of 2D sample vectors and categorical labels targets.
nb_classes = 2
nb_training_samples = 100
training_samples, training_categorical_labels = spiral_data(samples=nb_training_samples, classes=nb_classes)
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

# Adds another dimension to the array
# At first the array is an 1D array of ones and zeros
# With the reshape we leave the size of the first dimension as is by using -1 as the first argument
# The second(added) dimesion of the array is of size 1.
# So it's basically making an array of arrays but the second arrays only contain one value.
# Thus making the shape of the expected outputs match the shapes of the predicted outputs.
training_binary_labels = training_categorical_labels.reshape(-1, 1)
print('training_binary_labels.inputs_gadients.shape: ' + str(training_binary_labels.shape))


# Define the layers of the network and the function that will calculate the loss.
layer1 = Layer(2, 64, 0, logs_file=logs_file)
activation1 = Relu()
layer2 = Layer(64, 1, 0)
# last_activation_and_loss = Softmax_and_Categorical_loss()
activation2 = Sigmoid()
loss_function = BinaryCrossEntropy_loss()


# Training
optimizer = Adam_Optimizer(learning_rate=0.05, decay_rate=5e-7, logs_file=logs_file)
nb_epochs = 10001

def forward_pass(samples):
    # forward pass
    layer1.forward(samples)
    activation1.forward(layer1.outputs)
    layer2.forward(activation1.outputs)
    activation2.forward(layer2.outputs)


for epoch in range(nb_epochs): 
    forward_pass(training_samples)

    # backward pass: calculate gradients for the parameters and the inputs of each layer/activation function
    loss_function.backward(activation2.outputs, training_binary_labels)
    # print('loss_function.inputs_gadients.shape: ' + str(loss_function.inputs_gadients.shape))
    activation2.backward(loss_function.inputs_gadients)
    layer2.backward(activation2.inputs_gradients)
    activation1.backward(layer2.inputs_gradients)
    layer1.backward(activation1.inputs_gradients)

    # Optimization: apply negative of gradients 
    optimizer.pre_update_layers_params()

    optimizer.update_layer_params(layer1)
    optimizer.update_layer_params(layer2)

    optimizer.post_update_layer_params()

# Debugging

def calculate_mean_accuracy(expected_binary_labels):
    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary mask - array consisting of
    # True/False values, multiplying it by 1 changes it into array
    # of 1s and 0s
    predictions = (activation2.outputs > 0.5) * 1
    return np.mean(predictions == expected_binary_labels)

debug_str = 'nb training samples' + str(nb_training_samples) + '\n'
debug_str += 'Training loss: ' + str(loss_function.calculate_mean_loss(activation2.outputs, training_binary_labels)) + '\n'
debug_str += 'Training accuracy: ' + str(calculate_mean_accuracy(training_binary_labels)) + '\n'
# Measure loss and accuracy on test dataset
nb_test_samples = 100
test_samples, test_categorical_labels = spiral_data(samples=nb_test_samples, classes=nb_classes)
test_bianary_labels = test_categorical_labels.reshape(-1, 1)
forward_pass(test_samples)
debug_str += 'nb test samples: ' + str(nb_test_samples) + '\n'
debug_str += 'Test loss: ' + str(loss_function.calculate_mean_loss(activation2.outputs, training_binary_labels)) + '\n'
debug_str += 'Test accuracy: ' + str(calculate_mean_accuracy(test_bianary_labels)) + '\n'
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

forward_pass(test_samples)

# Convert NN output from confidence outputs to categorical outputes

# plot_samples(training_samples, test_samples, training_categorical_labels, bg_samples, categorical_bg_outputs, X, Y)