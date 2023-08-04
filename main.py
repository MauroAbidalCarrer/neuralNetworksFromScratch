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
from Model import Model
from MeanAccuracy_functions import calculate_mean_regression_accuracy

# logs_file for debugging
logs_file = open('logs.txt', '+a')

# Create datasets
nb_training_samples = 100
training_samples, expected_training_values = sine_data()

nb_test_samples = 100
test_samples, expected_test_values = sine_data()

# Create model
model = Model(
 layers=[
    Layer(1, 64, 0, activation_function=Relu()),
    Layer(64, 64, 1, activation_function=Relu()),
    Layer(64, 1, 2)
 ],
 loss_function=SquaredMean_Loss(),
 mean_accuracy_function=calculate_mean_regression_accuracy,
 optimizer=Adam_Optimizer(learning_rate=0.005, decay_rate=1e-3, logs_file=logs_file),
 logs_file=logs_file
)

# Training
model.train(training_samples=training_samples, training_expected_outputs=expected_training_values, epochs=10000)
model.debug_performances(
    training_inputs=training_samples, 
    expected_training_values=expected_training_values,
    test_inputs=test_samples,
    expected_test_values=expected_test_values
    )

# Debugging
logs_file.write('=======================================\n')
logs_file.close()