Why use the log function for the loss function instead of just using the difference between NN output and expected output?  
Why use Loss and accuracy?  
Why does the AdaGrad doesn't work if the update of the cache of the gradient is performed after the update of the parameters?  
    Because the first gradeitn gets multiplied by devided by epsilon instead of one (sqrt(squared(frist_gradient))) + epsilon?
What is the point of using epsilon then since the gradient_sum_caches will always be at least one?
Why, in SGD with momentum, is the gradient substracted by the momentum instead of being added?