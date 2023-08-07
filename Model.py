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

    def get_performances(self, inputs, expected_outputs):
        self.forward(inputs)
        accuracy = self.mean_accuracy_function(self.outputs, expected_outputs)
        loss = self.loss_function.calculate_mean_loss(self.outputs, expected_outputs)
        return accuracy, loss

    def get_performance_debug_str(self, inputs, expected_outputs, data_type):
        debug_str = ""
        self.forward(inputs)
        debug_str += f"{data_type} loss: {self.loss_function.calculate_mean_loss(self.outputs, expected_outputs):.4f}\n"
        debug_str += f"{data_type} accuracy: {self.mean_accuracy_function(self.outputs, expected_outputs):.4f} \n"
        return debug_str

    def write_final_performances_logs(self, training_inputs, expected_training_values, test_inputs, expected_test_values):
        if hasattr(self, "nb_epochs"):
            self.logs_file.write(f"nb training epochs: {self.nb_epochs}\n")
        self.logs_file.write(self.get_performance_debug_str(training_inputs, expected_training_values, "training"))
        self.logs_file.write(self.get_performance_debug_str(test_inputs, expected_test_values, "testing"))
        
    def train(self, inputs, expected_outputs, test_inputs, expected_test_values, *, epochs=10000, batch_size, perf_debug_interval=50):
        # Debugging variables
        self.nb_epochs = epochs # Save nb epochs for performaces logging.

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
                print(f"epoch: {epoch_index}")
                training_accuracy, training_loss = self.get_performances(inputs, expected_outputs)
                test_accuracy, test_loss = self.get_performances(test_inputs, expected_test_values)
                if hasattr(self, "last_training_accuracy"):
                    print(f"training loss: {training_loss:.4f} {(training_loss - self.last_training_loss):+.4f}")
                    print(f"training accuracy: {training_accuracy:.4f} {(training_accuracy - self.last_training_accuracy):+.4f}")
                    print(f"testing loss: {test_loss:.4f} {(test_loss - self.last_test_loss):+.4f}")
                    print(f"testing accuracy: {test_accuracy:.4f} {(test_accuracy - self.last_test_accuracy):+.4f}")
                else:
                    print(f"training loss: {training_loss:.4f}")
                    print(f"training accuracy: {training_accuracy:.4f}")
                    print(f"testing loss: {test_loss:.4f}")
                    print(f"testing accuracy: {test_accuracy:.4f}")
                self.last_training_loss = training_loss
                self.last_training_accuracy = training_accuracy
                self.last_test_loss = test_loss
                self.last_test_accuracy = test_accuracy
                print("")
        self.write_final_performances_logs(inputs, expected_outputs, test_inputs, expected_test_values)
