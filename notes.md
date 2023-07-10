  
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