import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, outputs_gradients):
        print('dense layer outputs_gradients shape: ', outputs_gradients.shape)
        # Gradients on parameters
        self.weights_gradients = np.dot(self.inputs.T, outputs_gradients)
        print('self.inputs.T[:, :3]:\n', self.inputs.T[:, :3])
        print('output_gradients[:3]:\n', outputs_gradients[3])
        print()
        self.biases_gradients = np.sum(outputs_gradients, axis=0, keepdims=True)
        # Gradient on values
        self.inputs_gradients = np.dot(outputs_gradients, self.weights.T)


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):

        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, outputs_gradients):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.inputs_gradients = outputs_gradients.copy()

        # Zero gradient where input values were negative
        self.inputs_gradients[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, outputs_gradients):

        # Create uninitialized array
        self.inputs_gradients = np.empty_like(outputs_gradients)

        # Enumerate outputs and gradients
        for index, (single_output, single_outputs_gradients) in \
                enumerate(zip(self.output, outputs_gradients)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.inputs_gradients[index] = np.dot(jacobian_matrix,
                                         single_outputs_gradients)


# Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, categorical_labels):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(categorical_labels.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                categorical_labels
            ]


        # Mask values - only for one-hot encoded labels
        elif len(categorical_labels.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * categorical_labels,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, outputs_gradients, categorical_labels):

        # Number of samples
        samples = len(outputs_gradients)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(outputs_gradients[0])

        # If labels are sparse, turn them into one-hot vector
        if len(categorical_labels.shape) == 1:
            categorical_labels = np.eye(labels)[categorical_labels]

        # Calculate gradient
        self.inputs_gradients = -categorical_labels / outputs_gradients
        # Normalize gradient
        self.inputs_gradients = self.inputs_gradients / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, categorical_labels):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, categorical_labels)


    # Backward pass
    def backward(self, outputs_gradients, categorical_labels):

        # Number of samples
        samples = len(outputs_gradients)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(categorical_labels.shape) == 2:
            categorical_labels = np.argmax(categorical_labels, axis=1)

        # Copy so we can safely modify
        self.inputs_gradients = outputs_gradients.copy()
        # Calculate gradient
        self.inputs_gradients[range(samples), categorical_labels] -= 1
        # Normalize gradient
        self.inputs_gradients = self.inputs_gradients / samples


# Create dataset
X, categorical_labels = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, categorical_labels)
# Let's see output of the first few samples:
# print('outputs:', loss_activation.output[:5])

# Print loss value
# print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(categorical_labels.shape) == 2:
    categorical_labels = np.argmax(categorical_labels, axis=1)
accuracy = np.mean(predictions==categorical_labels)

# Backward pass

loss_activation.backward(loss_activation.output, categorical_labels)
print("loss_activation.inputs_gradients.shape: ", loss_activation.inputs_gradients.shape)
dense2.backward(loss_activation.inputs_gradients)
activation1.backward(dense2.inputs_gradients)
dense1.backward(activation1.inputs_gradients)

# Print gradients
print('gradients:')
print('\nloss and activation gradient:\n', loss_activation.inputs_gradients[:5])
print('\nlayer1 weights gradient:\n', dense1.weights_gradients)
print('\nlayer1 biases gradient:\n', dense1.biases_gradients)
print('\nlayer2 weights gradient:\n', dense2.weights_gradients)
print('\nlayer2 biases gradient:\n', dense2.biases_gradients)