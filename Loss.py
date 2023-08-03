import numpy as np
# import * from Const

class Loss:
    def calculate_average_loss(self, nn_outputs, target_output):
        sample_losses = self.calculate(nn_outputs, target_output)
        return np.mean(sample_losses)


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

# I sure do like does esoteric terms...
class Binary_Categorical_cross_entropy_loss(Loss):

    def calculate_loss(self, nn_outputs, expected_outputs):
        clipped_expected_outputs = np.clip(expected_outputs, )