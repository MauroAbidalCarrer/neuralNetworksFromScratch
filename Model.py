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

    def perform_learning_step(self, inputs_batch, expected_outputs_batch):
        # Calculate gradients.
        self.forward(inputs_batch)
        # print('Output: ' + str(self.outputs))
        self.backward(expected_outputs_batch)
        # Optimize the model.
        self.optimizer.pre_update_layers_params()
        for i in range(self.layers_len):
            self.optimizer.update_layer_params(self.layers[i])
        self.optimizer.post_update_layer_params()

    def train(self, inputs, expected_outputs, test_inputs, expected_test_values, *, epochs=10000, batch_size, perf_debug_interval=50):
        nb_samples = len(inputs)
        remainder_batch_size = nb_samples % batch_size
        for epoch_index in range(epochs + 1):
            for batch_index in range(nb_samples // batch_size):
                # Slice inputs and expected outputs in batches
                inputs_batch = inputs[batch_index * batch_size:(batch_index + 1) * batch_size]
                expected_outputs_batch = expected_outputs[batch_index * batch_size:(batch_index + 1) * batch_size]
                # Perform
                self.perform_learning_step(inputs_batch, expected_outputs_batch)

            # In case there is a remainder of samples to train on:
            if remainder_batch_size:
                last_inputs_batch = inputs[nb_samples - remainder_batch_size:nb_samples]
                last_expected_outputs_batch = expected_outputs[nb_samples - remainder_batch_size:nb_samples]
                self.perform_learning_step(last_inputs_batch, last_expected_outputs_batch)

            # Debugging
            if perf_debug_interval != 0 and epoch_index % perf_debug_interval == 0:
                print("epoch", epoch_index)
                self.debug_performances(inputs, expected_outputs, test_inputs, expected_test_values)
                print("")
                

    def debug_performances(self, training_inputs, expected_training_values, test_inputs, expected_test_values):
        # Debug training data performances.
        debug_str = ""
        self.forward(training_inputs)
        # debug_str += 'nb training samples: ' + str(len(training_inputs)) + '\n'
        debug_str += f"training loss: {self.loss_function.calculate_mean_loss(self.outputs, expected_training_values):.2f}\n"
        debug_str += f"training accuracy: {self.mean_accuracy_function(self.outputs, expected_training_values):.2f} \n"
        # Debug test data performances.
        self.forward(test_inputs)
        # debug_str += 'nb test samples: ' + str(len(test_inputs)) + '\n'
        debug_str += f"testing loss: {self.loss_function.calculate_mean_loss(self.outputs, expected_test_values):.2f}\n"
        debug_str += f"testing accuracy: {self.mean_accuracy_function(self.outputs, expected_test_values):.2f}\n"
        print(debug_str, end='')
        # if self.logs_file:
        #     self.logs_file.write(debug_str)