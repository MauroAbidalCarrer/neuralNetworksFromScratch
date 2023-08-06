import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from activation_functions import *
from Layer import Layer
from Softmax_And_Categroical_Loss import *
from AdaptiveMomentum import *
from Model import Model
from MeanAccuracy_functions import calculate_mean_regression_accuracy
from zipfile import ZipFile
import os
import urllib
import urllib.request

# logs_file for debugging
logs_file = open('logs.txt', '+a')


# Download dataset if it's not already there.
if not os.path.isfile(ZIP_FILE):
    print(f'Downloading {FASHION_MNIST_DOWNLOAD_URL} and saving as {ZIP_FILE}...')
    urllib.request.urlretrieve(FASHION_MNIST_DOWNLOAD_URL, ZIP_FILE)

print('Unzipping images...')
with ZipFile(ZIP_FILE) as zip_images:
    zip_images.extractall(FOLDER)
print('Done!')

# Create datasets


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
# model.train(training_samples, expected_training_values, epochs=10000)
# model.debug_performances(training_samples, expected_training_values, test_samples, expected_test_values)

# Debugging
logs_file.write('=======================================\n')
logs_file.close()