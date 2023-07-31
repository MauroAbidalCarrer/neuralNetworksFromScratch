import numpy as np

class Loss:
    def calculate_average_loss(self, nn_outputs, target_output):
        sample_losses = self.calculate(nn_outputs, target_output)
        return np.mean(sample_losses)
    
    def calculate_regulirazation_loss(layer):
        regulirazation_loss =  layer.L1_weight_multiplier * np.sum(np.abs(layer.weights))
        regulirazation_loss += layer.L1_biases_multiplier * np.sum(np.abs(layer.biases))
        regulirazation_loss += layer.L2_biases_multiplier * np.sum(layer.weights ** 2)
        regulirazation_loss += layer.L2_biases_multiplier * np.sum(layer.biases ** 2)
        return regulirazation_loss


class Categorical_cross_entropy_loss(Loss):
    # Takes in the NN's batch fo outputs and the batch of targeted outputs.
    # Targeted outputs must be an array of categorical labels(indexes of the correct class).
    def calculate_loss(self, nn_ouputs, categorical_labels):
        # We need to clip the outputs to be higher than zero so we don't divide by zero afterword.
        # Since we are going to claculate the average of the loss,
        # We also need to clip the max value from 1, down to 1 - 1e-7 so the average doesn't get affected.
        clipped_outputs = np.clip(nn_ouputs, 1e-7, 1 - 1e-7)
        # Get a vector of all the confidences outputed by the NN at the targeted output index.
        # print('nn_ouputs.shape: ', nn_ouputs.shape, ', categorical_labels: ', categorical_labels.shape)
        target_confidences = clipped_outputs[range(len(nn_ouputs)), categorical_labels] 
        self.losses = -np.log(target_confidences)
        return self.losses
