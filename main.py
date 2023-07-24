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

# plot_samples(samples, categorical_labels)

# Define the layers of the network and the function that will calculate the loss.
layer1 = Layer(2, 64)
activation1 = Relu()
layer2 = Layer(64, 3)
last_activation_and_loss = Softmax_and_Categorical_loss()



# Generate a grid of points (replace the ranges as necessary)
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
bg_samples = np.array(np.c_[X.ravel(), Y.ravel()])

def draw_decision_boundary():
    #forward pass
    layer1.forward(bg_samples)
    activation1.forward(layer1.outputs)
    layer2.forward(activation1.outputs)
    last_activation_and_loss.activation.forward(layer2.outputs)

    one_hot_bg_outputs = last_activation_and_loss.activation.outputs
    categorical_bg_outputs = np.argmax(one_hot_bg_outputs, axis=1)
    # Round to nearest integer if your neural network outputs probabilities
    categorical_bg_outputs = np.round(categorical_bg_outputs).astype(int)
    # Plot the actual data and the background
    plot_samples(samples, categorical_labels, bg_samples, categorical_bg_outputs, X, Y)



logs_file = open('logs.txt', '+a')
# Training
optimizer = SGD_Optimizer(learning_rate=1, decay_rate=1e-3, momentum=0.03, logs_file=logs_file)
nb_epochs = 10001

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
    optimizer.pre_update_layers_params()

    optimizer.update_layer_params(layer1)
    optimizer.update_layer_params(layer2)

    optimizer.post_update_layer_params()
    

# Debugging
debug_str = 'Results:\n'
debug_str += 'Final loss:\t\t' + str(np.mean(last_activation_and_loss.loss.losses)) + '\n'
# Reduce the outputs to a one dimension array where we retain the index of the biggest confidence of each row.
categorical_NN_outputs = np.argmax(last_activation_and_loss.activation.outputs, axis=1)
# Makes an aray where each element is True if the guessed index matched the expected index, false otherwise.
succesfull_guesses = categorical_labels == categorical_NN_outputs
# np.mean of an array of booleans considers every True as one and every zero as false.
accuracy = np.mean(succesfull_guesses)
debug_str += 'Final accuracy:\t\t' + str(accuracy) + '\n'
print(debug_str, end="")
logs_file.write(debug_str)
logs_file.write('=======================================\n')
logs_file.close()

draw_decision_boundary()
