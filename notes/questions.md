Why use the log function for the loss function instead of just using the difference between NN output and expected output?  

## In ML, why do we use Loss and accuracy instead of just one of them?  
In machine learning, both loss (or cost) and accuracy are valuable metrics, but they serve different purposes and give insight into different aspects of the model's performance. Here's why we often consider both:

1. **Nature of the Metrics**:
    - **Loss**: This quantifies how well the model's predictions match the true labels. It's a continuous value, which makes it suitable for optimization. Most optimization algorithms (like gradient descent) require a differentiable function to minimize, and the loss function provides this.
    - **Accuracy**: This measures the proportion of correctly classified instances out of the total instances. It's a discrete metric, which can make it difficult (or impossible) to optimize directly.

2. **Granularity of Feedback**:
    - **Loss**: Provides detailed feedback on the correctness of the model's predictions. Two models can have the same accuracy but different loss values.
    - **Accuracy**: Gives a high-level understanding of how often the model is correct, but doesn't detail by how much the predictions are off.

3. **Training vs. Interpretability**:
    - **Loss**: Used during training to adjust the model's weights. Most ML training algorithms focus on minimizing the loss.
    - **Accuracy**: More interpretable for humans. If you tell someone your model has an accuracy of 95%, it's immediately clear that the model makes the correct prediction 95% of the time. It's a metric that's easy to understand, even for those not deeply familiar with ML.

4. **Limitations**:
    - **Loss**: Even if the loss is decreasing, it doesn't always mean the model's predictions are becoming more accurate. For instance, in regression tasks, a model might decrease the loss by improving its predictions on already well-predicted examples, while still failing on challenging examples.
    - **Accuracy**: Can be misleading, especially in imbalanced datasets. If you have a dataset where 99% of samples belong to Class A and only 1% belong to Class B, a naive model predicting always Class A will have an accuracy of 99% but is not really useful.

5. **Use Cases**:
    - **Loss**: Important for tasks like regression, where accuracy isn't a relevant metric. In regression, we often use metrics like Mean Squared Error (MSE) to quantify the model's performance.
    - **Accuracy**: Commonly used for classification tasks, especially for binary classification.

In summary, while there's some overlap between what loss and accuracy can tell you about a model's performance, they offer different perspectives. The loss provides a granular, continuous measure that's essential for model optimization, while accuracy offers a more general, interpretable assessment of how often the model makes correct predictions. It's often useful to consider both to get a comprehensive understanding of a model's performance.

---
Why does the AdaGrad doesn't work if the update of the cache of the gradient is performed after the update of the parameters?  
    Because the first gradeitn gets multiplied by devided by epsilon instead of one (sqrt(squared(frist_gradient))) + epsilon?
What is the point of using epsilon then since the gradient_sum_caches will always be at least one?
Why, in SGD with momentum, is the gradient substracted by the momentum instead of being added?