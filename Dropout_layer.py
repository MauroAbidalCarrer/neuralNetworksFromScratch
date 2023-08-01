import numpy as np
import nnfs
# Sets the random seed to 0 and does some other stuff to make the output repetable...
nnfs.init()

# class Dropout_Layer:

#     def __init__(self, size, dropout_fraction, debug_layer_index, logs_file=None):
#         self.size = size
#         self.dropout_fraction = dropout_fraction
#         self.nb_dropedout_neurons = round(dropout_fraction * size)
#         if logs_file:
#             logs_file.write('Dropout layer, index ' + str(debug_layer_index) + ', dropout_fraction: ' + str(dropout_fraction) + '\n')

#     def forward(self, inputs_batch):
#         self.mask = np.ones(self.size) / (1 - self.dropout_fraction)
#         zero_indices = np.random.choice(self.size, self.nb_dropedout_neurons, replace=False)
#         self.mask[zero_indices] = 0
#         self.outputs_batch = inputs_batch * self.mask
#         # print(self.mask)

#     def backward(self, gradients_batch):
#         self.inputs_gradients_batch = gradients_batch * self.mask

class Dropout_Layer:

    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
