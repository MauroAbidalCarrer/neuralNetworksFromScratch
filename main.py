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
nb_classes = 3
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
one_hot_targets = np.eye(nb_classes)[training_categorical_labels]

# Define the layers of the network and the function that will calculate the loss.
layer1 = Layer(2, 64, 0, logs_file=logs_file)
activation1 = Relu()
layer2 = Layer(64, 3, 0)
last_activation_and_loss = Softmax_and_Categorical_loss()



# Training
optimizer = Adam_Optimizer(learning_rate=0.05, decay_rate=5e-7, logs_file=logs_file)
nb_epochs = 10001

def forward_pass(inputs, categorical_labels):
    # forward pass
    layer1.forward(inputs)
    activation1.forward(layer1.outputs)
    layer2.forward(activation1.outputs)
    last_activation_and_loss.forward(layer2.outputs, categorical_labels)


for epoch in range(nb_epochs): 
    forward_pass(training_samples, training_categorical_labels)

    # backward pass
    last_activation_and_loss.backward(last_activation_and_loss.activation.outputs, training_categorical_labels)
    layer2.backward(last_activation_and_loss.input_gradients)
    activation1.backward(layer2.inputs_gradients)
    layer1.backward(activation1.inputs_gradients)

    # Optimization
    optimizer.pre_update_layers_params()

    optimizer.update_layer_params(layer1)
    optimizer.update_layer_params(layer2)

    optimizer.post_update_layer_params()

# Debugging
def calculate_mean_loss():
    return np.mean(last_activation_and_loss.loss.losses)

def calculate_mean_accuracy(expected_categorical_labels):
    categorical_NN_outputs = np.argmax(last_activation_and_loss.activation.outputs, axis=1)
    succesfull_guesses = expected_categorical_labels == categorical_NN_outputs
    return np.mean(succesfull_guesses)

debug_str = 'nb training samples' + str(nb_training_samples) + '\n'
debug_str += 'Training loss: ' + str(calculate_mean_loss()) + '\n'
debug_str += 'Training accuracy: ' + str(calculate_mean_accuracy(training_categorical_labels)) + '\n'
# Measure loss and accuracy on test dataset
nb_test_samples = 100
test_samples, test_categorical_labels = spiral_data(samples=nb_test_samples, classes=nb_classes)
forward_pass(test_samples, test_categorical_labels)
debug_str += 'nb test samples: ' + str(nb_test_samples) + '\n'
debug_str += 'Test loss: ' + str(calculate_mean_loss()) + '\n'
debug_str += 'Test accuracy: ' + str(calculate_mean_accuracy(test_categorical_labels)) + '\n'
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
#forward pass
layer1.forward(bg_samples)
activation1.forward(layer1.outputs)
layer2.forward(activation1.outputs)
last_activation_and_loss.activation.forward(layer2.outputs)
one_hot_bg_outputs = last_activation_and_loss.activation.outputs

# Convert NN output from confidence outputs to categorical outputes
categorical_bg_outputs = np.argmax(one_hot_bg_outputs, axis=1)
categorical_bg_outputs = np.round(categorical_bg_outputs).astype(int)

plot_samples(training_samples, test_samples, training_categorical_labels, bg_samples, categorical_bg_outputs, X, Y)
