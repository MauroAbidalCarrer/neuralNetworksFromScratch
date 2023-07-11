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
neuron_output = relu(sum(bias, w0, i0, w1, i1, w2, i2))
neuron_output = 6

# derivative value of the loss function w.r.t the neuron output
# (i.e if we increased the output by one, the loss would increase by one)
d_neuron_output = 1

# derivative value of the loss function w.r.t w0
d_w0 =  relu`(sum(bias, mul(w0, i0), mul(w1, i1), mul(w2, i2)) * 
        sum`(bias, mul(w0, i0), mul(w1, i1), mul(w2, i2) * 
        mul`(w0, i0) *
        d_neuron_output

# Now that we know how to calculate the derivative value of the loss function w.r.t w0, we just need to cacluate the dreivatives.  

# relu(x) = (x if x > 0 else 0) therefore:
relu`(x) = (1. if x > 0 else 0)

# The partial derivative function of the sum w.r.t one of its inputs is always 1.
sum`(x) = 1

# The partial derivative function of the multiplication w.r.t one of its input(a) is always equal to the other input(b).
mul`(x) = b

# Therefore we can simplify the derivative value of the loss function w.r.t w0 equation to:
d_w0 =  (1 if sum(bias, mul(w0, i0), mul(w1, i1), mul(w2, i2)) > 0 else 9) *
        1 *
        i0 *
        d_neuron_output

d_w0 = 1 * 1 * 1 * 1 # How convinient...
d_w0 = 1
```


Now that we know how to calculate the partial derivative of a weight, lets caculate the gradient of the weights.  


```python

weights = []
inputs = []
bias 

d_neuron_output

neuron_output = 6

d_weights = f

```