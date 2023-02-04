# Book Notes

## [Machine Learning Design Patterns](https://github.com/GoogleCloudPlatform/ml-design-patterns)

**Outlier clipping:** Given an interval, values outside the interval are clipped to the interval edges.

## [The fastai book](https://github.com/fastai/fastbook)

### Learning Rates

Smith Learning Rate Finder

Discriminative Learning Rates

### Loss Functions

- `nn.CrossEntropyLoss` for single-label classification
- `nn.BCEWithLogitsLoss` for multi-label classification
- `nn.MSELoss` for regression

Try tree ensembles for structured data, neural networks for unstructured data (text, images, speech).

### Bagging

Although each of the models trained on a subset of the data will make more errors than a model trained on all data, these errors are uncorrelated, and the average error goes to zero as the number of models increases.

Bagging thus is a particular approach to ensembling, or combining the results of multiple models together.

### NNs

[A neural network from the foundations](https://github.com/fastai/fastbook/blob/master/17_foundations.ipynb)

### RNNs

RNNs suffer from vanishing / exploding gradients.

Float point representations

<https://course.fast.ai>

## [Hundred Page Machine Learning](http://ema.cri-info.cm/wp-content/uploads/2019/07/2019BurkovTheHundred-pageMachineLearning.pdf)

### SVM

#### Dealing with Noise

To extend SVM to cases in which the data is not linearly separable, we introduce the hinge loss function: $\max(0, 1 - y_i(\mathbf{wx_i}-b))$.

The hinge loss function is zero if the constraints (a) and (b) are satisified, in other words, if $\mathbf{wx_i}$ lies on the correct side of the decision boundary. For data on the wrong side of the decision boundary, the function's value is proportional to the distance from the decision boundary.

We then wish to minimise the following cost function:

$$
C \lVert \mathbf{w} \rVert^2 + \frac{1}{N} \sum_{i=1}^N{\max(0, 1 - y_i(\mathbf{wx_i}-b))}
$$

where the hyperparameter $C$ is the tradeoff between increasing the size of the decision boundary and ensuring that each $\mathbf{x_i}$ lies on the correct side of the decision boundary. The value of $C$ is usually chosen experimentally, just like ID3's hyperparameters $\epsilon$ and $d$. SVMs that optimise hinge loss are called *soft-margin* SVMs, while the original formulation is referred to as *hard-margin* SVM.

As you can see, for sufficiently high values of $C$, the second term in the cost function will become negligible, so the SVM algorithm will try to find the highest margin by completely ignoring misclassification. As we decrease the value of $C$, making classification errors is becoming more costly, so the SVM algorithm will try to make fewer mistakes by sacrificing the margin size. As we have already discussed, a larger margin is better for generalisation. Therefore, $C$ regulates the tradeoff between classifying the training data well (minimising empirical risk) and classifying future examples well (generalisation).

The only difference between a linear regression model and SVM is the SVM's sign operator. The hyperplane in the SVM plays the role of the decision boundary: it's used to separate two groups of examples from one another.

As such, it has to be as far from each group as possible. On the other hand, the hyperplane in linear regression is chosen to be as close to all training examples as possible.

In practice, L1 regularisation produces a sparse model, a model that has most of its parameters equal to zero, provided the hyperparameter alpha is large enough. So L1 performs feature selection by deciding which features are essential for prediction and which are not. That can be useful if you want to increase model explainability.

However, if your goal is to maximise the performance of the model on the holdout data, then L2 usually gives better results. L2 also has the advantage of being differentiable, so gradient descent can be used for optimising the objective function.

L1 is called Lasso, L2 Ridge, and their combination is an Elastic Net.
