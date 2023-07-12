## Layer backpropagation

Given a loss function gradient relative to the layer's output, we need to calculate the:
- weights gradient
- input gradient
- bias gradient

How to calculate the weights gradient:
The dot product between a vector and a matrix needs to be broken down to two atomic :
- a mult between each row vector of the matrix and the input vector
- a sum of the components of the row vectors of the matrix product
So the gradient of the weights is a matrix where every row equals the input vector.

How to calculate the input gradient:
Just like for the weights, we break the dot product into two atomic operations.
However, how d 

How to calculate the bias gradient:
TL;DR: biases gradient = a ones vector