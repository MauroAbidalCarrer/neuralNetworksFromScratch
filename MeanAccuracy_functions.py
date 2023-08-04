import numpy as np


def calculate_mean_regression_accuracy(model_outputs, expected_outputs):
    # accuracy_margin defines the margin of error allowed for the nn_outputs.
    # It is relative to the standard deviation of the expected_outputs (np.std(expected_outputs)).
    # https://en.wikipedia.org/wiki/Standard_deviation
    # Standard deviation is a measure of how much variation there is in the expected outputs batch.
    # This is because we want the network to output data that is similar from the training data.
    accuracy_margin = np.std(expected_outputs) / 250
    return np.mean(np.abs(expected_outputs - model_outputs) <= accuracy_margin)