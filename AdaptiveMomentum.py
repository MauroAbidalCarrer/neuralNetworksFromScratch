import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()
from Constants import *

# Similar to AdaGrad, this optimizer uses an additional parameter 
class Adam_Optimizer:
    """
        Built on top of SGD with momentum and RMSP, this optimizer uses both momentum and a per-parameter learning.
        Hencce the name Adaptive momentumm, shorten to Adam. 
        In addition, Adam corrects the momentum and gradient_sum_cache during the beggining of the training.
    """
    def __init__(self, learning_rate=1, decay_rate=0., logs_file=None, gradient_sum_cache_lerp_param=0.001, momentum=0.3, momentum_lerp_param=0.01):
        """
        Args:
            weights_gradients_sum_cache (float, optional):
                Defines how much the gradient_sum_cache lerps toward the gradient squared.  
                Defaults to 0.999.  
                Must remain in the [0, 1] range.
                Corresponds to 1 - beta_2 attribute in the NNFS book.
            momentum_weight (float, optional):
                Defines how much of the gradient momentum lerps toward the last gradient.
                Defaults to 0.99.
                Must remain in the [0, 1] range.
                Corresponds to 1 - beta_1 attribute in the NNFS book.
        """
        
        self.initial_learning_rate = learning_rate  
        self.learning_rate = self.initial_learning_rate
        self.decay_rate = decay_rate
        self.nb_update_iterations = 0
        self.gradient_sum_cache_lerp_param = gradient_sum_cache_lerp_param
        self.momentum = momentum
        self.momentum_lerp_param = momentum_lerp_param
        # Debugging
        debug_str = 'Optimizer: ' + self.__class__.__name__ + '\n'
        debug_str += 'learning rate:\t\t' + str(learning_rate) + '\n'
        debug_str += 'learning_decay_rate:\t' + str(decay_rate) + '\n'
        debug_str += 'momentum_lerp_param:\t' + str(momentum_lerp_param) + '\n'
        debug_str += 'gradient_sum_cache_lerp_param:\t' + str(gradient_sum_cache_lerp_param) + '\n'
        print(debug_str, end="")
        if logs_file:
            logs_file.write(debug_str)

    def pre_update_layers_params(self):
        self.nb_update_iterations += 1
        if (self.decay_rate):
            # Don't replace the '1 +' by setting 'self.nb_update_iterations = 1' because it will get multiplied by decay rate.
            # We want the denominator to always be at least be 1 otherwise increase the value self.learning_rate instead of decreasing it.
            self.learning_rate = self.initial_learning_rate / (1 + self.decay_rate * self.nb_update_iterations)

    def update_layer_params(self, layer):
        # Add weights and biases updates sum cache if the layer doesn't already have it .
        if not hasattr(layer, 'weights_gradients_sum_cache'):
            layer.weights_gradients_sum_cache = np.zeros_like(layer.weights_gradient)
            layer.weights_gradient_momentum = np.zeros_like(layer.weights_gradient)
            layer.biases_gradients_sum_cache = np.zeros_like(layer.biases_gradient)
            layer.biases_gradient_momemtum = np.zeros_like(layer.biases_gradient)

        # Update gradient momentums
        layer.weights_gradient_momentum = lerp(layer.weights_gradient_momentum, layer.weights_gradient, self.momentum_lerp_param)
        layer.biases_gradient_momemtum = lerp(layer.biases_gradient_momemtum, layer.biases_gradient, self.momentum_lerp_param)
        corrected_weights_gradient_momentum = self.correct_array(layer.weights_gradient_momentum, self.momentum_lerp_param)
        corrected_biases_gradient_momemtum = self.correct_array(layer.biases_gradient_momemtum, self.momentum_lerp_param)
            
        # Update gradient sum cache.
        # Step1: multiply the caches by the weights_gradients_sum_cache, effectively keeping only a fraction of it.
        # layer.weights_gradients_sum_cache = layer.weights_gradients_sum_cache * self.gradient_sum_cache_lerp_param
        # layer.biases_gradients_sum_cache = layer.biases_gradients_sum_cache * self.gradient_sum_cache_lerp_param
        # # Step2: add the new last gradient to the sum by 1 - weights_gradients_sum_cache, effectively adding only a fraction of it tp the sum.
        # # We update the squared root of the gradient and to get the absolute value of the gradient AND to make the cache grow slower(when getting the squared root) than if we were to simply add the absolute values.
        # layer.weights_gradients_sum_cache += (1 - self.gradient_sum_cache_lerp_param) * layer.weights_gradient ** 2 
        # layer.biases_gradients_sum_cache += (1 - self.gradient_sum_cache_lerp_param) * layer.biases_gradient ** 2
        layer.weights_gradients_sum_cache = lerp(layer.weights_gradients_sum_cache, layer.weights_gradient ** 2, self.gradient_sum_cache_lerp_param)
        layer.biases_gradients_sum_cache = lerp(layer.biases_gradients_sum_cache, layer.biases_gradient ** 2, self.gradient_sum_cache_lerp_param)
        
        # corrected_weights_gradients_sum_cache = layer.weights_gradients_sum_cache / (1 - layer.weights_gradients_sum_cache ** (self.nb_update_iterations + 1))
        corrected_weights_gradients_sum_cache = self.correct_array(layer.weights_gradients_sum_cache, self.gradient_sum_cache_lerp_param)
        # corrected_biases_gradients_sum_cache = layer.biases_gradients_sum_cache / (1 - layer.weights_gradients_sum_cache ** (self.nb_update_iterations + 1))
        corrected_biases_gradients_sum_cache = self.correct_array(layer.biases_gradients_sum_cache, self.gradient_sum_cache_lerp_param)
        
        # Update layer parameters.SADFSDFASDASad
        layer.weights -= self.learning_rate * corrected_weights_gradient_momentum / (np.sqrt(corrected_weights_gradients_sum_cache) + small_value)
        layer.biases -= self.learning_rate * corrected_biases_gradient_momemtum / (np.sqrt(corrected_biases_gradients_sum_cache) + small_value)

    def correct_array(self, np_array, lerp_param):
        """
            Calculate the corrected tensor(either gradient momentum or sum cache) by dividing it by (1 - weights_gradients_sum_cache ** (nb_iterations + 1)).
            The result of this expression wil be high during the first learning steps and will decrease over time.
            This addresses the issue of small gradient cache and momentum during the beggining of the training.
        """
        return np_array / (1 - (1 - lerp_param) ** (self.nb_update_iterations + 1))

    def post_update_layer_params(self):
        self.nb_update_iterations += 1

def lerp(a, b, t):
    return b * t + (1 - t) * a