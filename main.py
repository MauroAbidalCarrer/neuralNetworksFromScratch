import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable
nnfs.init()
from activation_functions import *
from Layer import Layer
from Softmax_And_Categroical_Loss import *
from AdaptiveMomentum import *
from Model import Model
from MeanAccuracy_functions import *
from zipfile import ZipFile
import os
import urllib
import urllib.request
import cv2

# logs_file for debugging
logs_file = open('logs.txt', '+a')
logs_file.write('=======================================\n')

# Download dataset if it's not already there.
if not os.path.isfile(ZIP_FILE):
    print(f"Downloading {FASHION_MNIST_DOWNLOAD_URL} and saving as {ZIP_FILE}...", end="", flush=True)
    urllib.request.urlretrieve(FASHION_MNIST_DOWNLOAD_URL, ZIP_FILE)
    print('Done!')

# Extract the .zip if there is no fashion_mnist_images folder.
if not os.path.isdir('fashion_mnist_images'):
    print('Unzipping images...', end="", flush=True)
    with ZipFile(ZIP_FILE) as zip_images:
        zip_images.extractall(FOLDER)
        print('Done!')

# Load datasets.
def load_dataset(dataset_type):
    """Type must be either 'train' or 'test'."""
    print(f"Loading {dataset_type}ing dataset...", flush=True)
    inputs = []
    expected_outputs = []
    for label_class in os.listdir(os.path.join(SMALL_FOLDER, dataset_type)):
        folder_path = os.path.join(SMALL_FOLDER, dataset_type, label_class)
        print(f"loading {folder_path}...")
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # print('Loading ' + image_path)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            inputs.append(image)
            expected_outputs.append(label_class)
    # Set the range of the values from [0, 255] to [-1, 1].
    inputs = np.array(inputs).astype('float32') / 127.5 - 1 
    # imread returns a matrix of values for each image but we want a vector for each image.
    # So we flatten the array, setting the shape from (nb_samples, side_length, side_length) to (nb_samples, side_length * side_length).
    # inputs.reshape(inputs.shape[0], -1) preservs the first dimension's size and flattens the remaining ones.
    inputs = inputs.reshape(inputs.shape[0], -1)
    expected_outputs = np.array(expected_outputs).astype('uint8').reshape(-1)
    # Shuffle datasets, this is to prevent the model from being biased toward predicting a single class during 
    shuffle_indices = np.array(range(inputs.shape[0]))
    np.random.shuffle(shuffle_indices)
    inputs = inputs[shuffle_indices]
    print('done.')
    return (inputs, expected_outputs[shuffle_indices])

training_inputs, expected_training_outputs = load_dataset('train')
test_inputs, expected_test_outputs = load_dataset('test')

# Preprocess datasets.


# Create model
model = Model(
 layers=[
    Layer(INPUT_VECTOR_SIZE, 64, 0, L2_biases_multiplier=0.01, L2_weights_multiplier=0.01, activation_function=Relu()),
    Layer(64, 64, 1, L2_biases_multiplier=0.01, L2_weights_multiplier=0.01, activation_function=Relu()),
    Layer(64, 10, 2, L2_biases_multiplier=0.01, L2_weights_multiplier=0.01, activation_function=Softmax_and_Categorical_loss())
 ],
 loss_function=Softmax_and_Categorical_loss(),
 mean_accuracy_function=calculate_mean_classification_accuracy,
 optimizer=Adam_Optimizer(learning_rate=0.001, decay_rate=0.0005, logs_file=logs_file),
 logs_file=logs_file
)

# Training
model.train(training_inputs, expected_training_outputs, test_inputs, expected_test_outputs, epochs=3000, batch_size=300, perf_debug_interval=200)

logs_file.close()