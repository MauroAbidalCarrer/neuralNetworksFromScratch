import numpy as np
from Layer import Layer
from Model import Model

class Model_beta(Model):

    def __init__(self, *, layers, loss_function, mean_accuracy_function, optimizer, logs_file):
        super().__init__(layers=layers, 
                         loss_function=loss_function, 
                         mean_accuracy_function=mean_accuracy_function, 
                         optimizer=optimizer, 
                         logs_file=logs_file)

        logs_file.write('-----BETA MODEL-----\n')

    def perform_learning_step_beta(self, inputs, expected_outputs):
        # Calculate gradients.
        self.forward(inputs)
        # print('Output: ' + str(self.outputs))
        # self.backward(expected_outputs_batch)
        gradients = self.loss_function.backward(self.outputs, expected_outputs)
        # Optimize the model.
        self.optimizer.pre_update_layers_params()
        for i in range(self.layers_len - 1, -1, -1):
            self.layers[i].calculate_params_gradients(gradients)
            self.optimizer.update_layer_params(self.layers[i])
            gradients = self.layers[i].calculate_inputs_gradients(gradients)
        self.optimizer.post_update_layer_params()

    
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
                self.perform_learning_step_beta(inputs_batch, expected_outputs_batch)

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