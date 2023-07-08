import numpy as np

class Loss:
    def calculate_average_loss(self, nn_outputs, target_output):
        sample_losses = self.calculate(nn_outputs, target_output)
        return np.mean(sample_losses)


class Categorical_cross_entropy_loss(Loss):
    
    # Takes in the NN's batch fo outputs and the batch of targeted outputs.
    # Targeted outputs must be an array of categorical labels(indexes of the correct class).
    def calculate_loss(self, nn_ouputs, target_outputs):
        # We need to clip the outputs to be higher than zero so we don't divide by zero afterword.
        # Since we are going to claculate the average of the loss,
        # we also need to clip the max value from 1, down to 1 - 1e-7 so the average doesn't get affected.
        clipped_outputs = np.clip(nn_ouputs, 1e-7, 1 - 1e-7)                        
        # Get a vector of all the confidences outputed by the NN of the targeted output.
        target_confidences = clipped_outputs[range(len(nn_ouputs)), target_outputs] 
        self.loss = -np.log(target_confidences)
        return self.loss
    
    # def calculate_gradient_of_loss_wrt_nn_output(self, one_hot_targets):
    #     self.gradient = one_hot_targets / 
