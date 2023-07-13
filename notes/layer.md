# Forward pass

The entire neural network is implemented to work with batches of inputs and outputs.  
To perform a forward pass with a batch of inputs without using a loop, we take a matrix as input instead of a vector.  
In the matrix each row is a sample/input.  
To perform a dot product on each sample of the batch and the weight matrix without using a loop, we transpose the weight matrix.
This way we can perform the dot product on the two matrices.  
The reason for transposing the weight matrixm is that the dot product between the vectors of the two matrices,  
is performed between each row of the first matrix(in our case, the samples) and each column of the second matrix(in our case, the weights).  
So we need each column of the weight matrix to be the weights of a single neuron.

## Layer backpropagation

Given a loss function gradient relative to the layer's output, we need to calculate the:
- weights gradient
- input gradient
- bias gradient

How to calculate the weights gradient:  
The dot product between a vector and a matrix needs to be broken down to two atomic:  
- a mult between each row vector of the matrix and the input vector  
- a sum of the components of the row vectors of the matrix product  

So the gradient of the weights is a matrix where every row equals the input vector multiplied by the output gradient activation function.  
Given that our weight matrix is transposed (the matrix is of shape input_size x neuron_size) to work with batches of samples,  
The matrix gradient will actually be a matrix where every column is the input vector multiplied by the output gradient activation function.  

How to calculate the input gradient:  
Just like for the weights, we break the dot product into two atomic operations.  
We end up with a matrix gradient that equals the weights matrix.  
However, the input gradient needs to be a vector.  
So we can sum the matrix on its columns since each column of this gradient matrix is relative to the same input component.  
Given that our weight matrix is transposed (the matrix is of shape input_size x neuron_size) to work with batches of samples,  
we actually need to sum the matrix on its rows(i believe...)  

How to calculate the bias gradient:
TL;DR: biases gradient = a ones vector