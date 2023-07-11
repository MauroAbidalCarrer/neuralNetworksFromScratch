## Layer backpropagation

Given a loss function gradient relative to the layer's output, we need to calculate the:
- weights gradient
- input gradient
- bias gradient

```python
d_weights = activation`(sum(dot(weights, inputs), baises)) *
            sum`(dot(weights, inputs), baises) *
            dot`(weights, inputs) *
            d_output
```