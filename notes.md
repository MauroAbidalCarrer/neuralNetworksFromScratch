## Softmax gradient  
  
Because the SoftMax takes in a vector and outputs a vector where each component of the output is dependant on each input,   
the gradient of the SoftMax is a matrix.  
More specifically, a Jacobian Matrix.(I really hope I don't forget all of this in a month...)  
Where each row corresponds to a particular input and each columm corresponds to a particular output.  
  
This poses an issue as we can't back propagate this gradient in the neurons that act as the input of the Softmax.  
However, we are going to have to apply the chain rule on this gradient by the previous one, i.e multiply them together, which is a vector.  
And since the result of the product of a vector and a matrix is a vector the problem kind of solves itself.  
In this case we are using the dot product, idk we are using the dot product instead of multiplying the vector and the matrix.  
  
> L = vector size  
> i = sample index  
> j = output/neuron index that we want to calculate the derivative of  
> k = input sample index that we want to calculate the derivative with respect to  

## Neuron backpropagation(chapter 9)

We have a derivative value of the loss function w.r.t the output of the neuron.  
- Goal1: Calculate the gradient of the parameters.
- Goal2: Calculate the gradient of the inputs, so that we can continue the back propagation on the previous neurons.  

Steps to caculate the derivative value of a certain parameter of input:
- Calculate the derivatives values(and therfore also the derivative functions) of all the atomic operations(sums and multiplications) w.r.t their respective inputs.  
- Multiply them together with the derivative value of the loss function w.r.t to the neuron ouput to get the derivative value of the loss function w.r.t a certain parameter or input.

```python
# weights
w0 = -3
w1 = -1
w2 = 2 
# inputs
i0 = 1
i1 = -2
i2 = 3

bias = 1

# neuron output written as a series of atomic operations
neuron_output = relu(bias + (w0 * i0) + (w1 * i1) + (w2 * i2))

#derivative of the loss function w.r.t the neuron output
d_neuron_output = 1

#derivative of the loss function w.r.t w0
d_w0 = relu`(bias + (w0 * i0) + (w1 * i1) + (w2 * i2))
```