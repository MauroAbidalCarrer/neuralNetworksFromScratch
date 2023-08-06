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
import cv2

# logs_file for debugging
logs_file = open('logs.txt', '+a')


# Download dataset if it's not already there.
if not os.path.isfile(ZIP_FILE):
    print('Downloading ' + FASHION_MNIST_DOWNLOAD_URL + ' and saving as ' + ZIP_FILE + '...', end="", flush=True)
    urllib.request.urlretrieve(FASHION_MNIST_DOWNLOAD_URL, ZIP_FILE)
    print('Done!')

print('yes: ' + FASHION_MNIST_DOWNLOAD_URL)
# Extract the .zip if there is no fashion_mnist_images folder.
if not os.path.isdir('fashion_mnist_images'):
    print('Unzipping images...', end="", flush=True)
    with ZipFile(ZIP_FILE) as zip_images:
        zip_images.extractall(FOLDER)
        print('Done!')

# Load datasets.
def load_dataset(dataset_type):
    """Type must be either 'train' or 'test'."""
    print('Loading ' + dataset_type + 'ing dataset... ', end="", flush=True)
    inputs = []
    expected_outputs = []
    for label_class in os.listdir(os.path.join(FOLDER, dataset_type)):
        folder_path = os.path.join(FOLDER, dataset_type, label_class)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # print('Loading ' + image_path)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            inputs.append(image)
            expected_outputs.append(label_class)
    inputs = np.array(inputs).astype('float32') / 255. - 0.5
    # print('Going to return')
    expected_outputs = np.array(expected_outputs)
    print('done!')
    return (inputs, expected_outputs.astype('uint8'))

training_inputs, training_outputs = load_dataset('train')
test_inputs, test_outputs = load_dataset('test')


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