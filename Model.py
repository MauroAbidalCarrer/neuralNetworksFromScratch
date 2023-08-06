import numpy as np
from Layer import Layer

class Model:

    def __init__(self, *, layers, loss_function, mean_accuracy_function, optimizer, logs_file=None):
        self.layers = layers
        self.layers_len = len(layers)
        self.loss_function = loss_function
        self.mean_accuracy_function = mean_accuracy_function
        self.optimizer = optimizer
        self.logs_file = logs_file

    def forward(self, inputs):
        data_representations = inputs
        for i in range(self.layers_len):
            data_representations = self.layers[i].forward(data_representations)
        self.outputs = data_representations

    def backward(self, expected_outputs):
        gradients = self.loss_function.backward(self.outputs, expected_outputs)
        for i in range(self.layers_len - 1, -1, -1):
            gradients = self.layers[i].backward(gradients)

    def train(self, training_samples, training_expected_outputs, *, epochs=10000, batch_size):
        nb_samples = len(training_samples)
        for _ in range(epochs + 1):
            training_batch = nb_samples
            for _ in range(batch_size):
                self.forward(training_samples)
                self.backward(training_expected_outputs)
                
                self.optimizer.pre_update_layers_params()
                for i in range(self.layers_len):
                    self.optimizer.update_layer_params(self.layers[i])
                self.optimizer.post_update_layer_params()
            if 

    def debug_performances(self, training_inputs, expected_training_values, test_inputs, expected_test_values):
        # Debug training data performances.
        debug_str = ""
        debug_str += 'nb training samples: ' + str(len(training_inputs)) + '\n'
        debug_str += 'training loss: ' + str(self.loss_function.calculate_mean_loss(self.outputs, expected_training_values)) + '\n'
        debug_str += 'training accuracy: ' + str(self.mean_accuracy_function(self.outputs, expected_training_values)) + '\n'
        # Debug test data performances.
        self.forward(test_inputs)
        debug_str += 'nb test samples: ' + str(len(test_inputs)) + '\n'
        debug_str += 'accuracy loss: ' + str(self.loss_function.calculate_mean_loss(self.outputs, expected_test_values)) + '\n'
        debug_str += 'accuracy accuracy: ' + str(self.mean_accuracy_function(self.outputs, expected_test_values)) + '\n'
        print(debug_str, end='')
        if self.logs_file:
            self.logs_file.write(debug_str)