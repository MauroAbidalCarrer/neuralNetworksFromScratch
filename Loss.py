import numpy as np
# import * from Const

class Loss:
    def calculate_mean_loss(self, nn_outputs, target_output):
        sample_losses = self.calculate_loss(nn_outputs, target_output)
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
class BinaryCrossEntropy_loss(Loss):
    # Forward pass
    def calculate_loss(self, nn_outputs, expected_outputs):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        nn_outputs_clipped = np.clip(nn_outputs, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(expected_outputs * np.log(nn_outputs_clipped) +
        (1 - expected_outputs) * np.log(1 - nn_outputs_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses
        # Backward pass

    def backward(self, nn_outputs, expected_outputs):
        # print('nn_outputs.shape: ' + str(nn_outputs.shape))
        # print('expected_outputs.shape: ' + str(expected_outputs.shape))
        nn_outputs_len = len(nn_outputs)
        outputs_len = len(nn_outputs[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(nn_outputs, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.inputs_gadients = -(expected_outputs / clipped_dvalues -
        (1 - expected_outputs) / (1 - clipped_dvalues)) / outputs_len
        # Normalize gradient
        self.inputs_gadients = self.inputs_gadients / nn_outputs_len

class SquaredMean_Loss(Loss):

    def calculate_loss(self, nn_outputs, expected_outputs):
        # Calculate loss
        sample_losses = np.mean((expected_outputs - nn_outputs)**2, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, nn_outputs, expected_outputs):
        nn_outputs_len = len(nn_outputs)
        output_len = len(nn_outputs[0])
        # Gradient on values
        self.inputs_gadients = -2 * (expected_outputs - nn_outputs) / output_len
        # Normalize gradient
        self.inputs_gadients = self.inputs_gadients / nn_outputs_len

        