# Improving Deep Neural Networks: Hyperparameter Tuning, Regularisation and Optimisation

Machine learning is an iterative process, finding the best number of layers, hidden units, learning rate, activation functions, etc.

## Train / Dev / Test Sets

*Make sure validation and test sets are sampled from the same distribution!*

## Bias and Variance

Bias refers to the training set error; variance to the generalisation error.

Examples:

- High variance (overfitting): low train set error, high test set error
- High bias (underfitting): high train and test set errors
- High bias and variance: high train set error and even higher test set error, i.e. when the model fails to fit the data AND to generalise

First, to reduce bias:

- Bigger network
  - Increase the number of units in each hidden layer
  - Make the network deeper
- Train longer

To reduce variance:

- More data
- Regularisation

Given, you can train larger neural networks and get more data, the bias-variance trade-off isn't really relevant anymore in the deep learning era.

## Regularisation

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m{L(\hat{y}^{(i)}, y^{(i)})} + \frac{\lambda}{2m} \lVert w \rVert _2^2
$$

**L2 Regularisation** uses Euclidean distance and is used more often:

$$
\lVert w \rVert _2^2 = \sum_{j=1}^{n_x}{w_j^2} = w^T w
$$

L1 Regularisation uses the Manhattan distance and results in a sparse matrix with a lot of zeros:

$$
\lVert w \rVert _1 = \sum_{j=1}^{n_x}{|w_j|}
$$

Both L1 and L2 are forms of **weight decay**: gradient descent shrinks the weights on every iteration.

**[Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/pdf/1711.00489.pdf)**

*The larger the value of the regularization parameter $\lambda$, the smaller the magnitude of the weights, staying close to the origin where the slope is steeper.* It is a hyperparameter that can be tuned on the validation set.

Neural networks can also be regularised using **dropout**: Can't rely on any one feature, so have to spread out weights.

During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if `keep_prob` is 0.5, then we will on average shut down half the nodes (**`1-keep_prob`**), so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when `keep_prob` is other values than 0.5.

Other regularisation methods include **data augmentation** and **early stopping**.

## Optimisation

### Input Normalisation

Easier and faster to optimise cost functions; **will converge faster** and may not need such a large learning rate.

### [Weight Initialisation to avoid Vanishing / Exploding Gradients](https://www.deeplearning.ai/ai-notes/initialization/index.html)

Different initializations lead to very different results: If $W > I$, the weights can increase exponentially; if $W < I$, the weights may decrease exponentially.

Random initialization is used to break symmetry and make sure different hidden units can learn different things.

*Make sure to initialise weights to values that aren't too large!*

As a rule of thumb, use

- $Var(w) = 1/n$ (**Xavier initialisation**)
- or $2/n$ (**He initialisation**) for a ReLU activation function.

```python
W = np.random.randn(shape) * np.sqrt(2./layers_dims[l-1])
```

- `numpy.random.rand()` produces numbers in a [uniform distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/rand.jpg).
- `numpy.random.randn()` produces numbers in a [normal distribution](https://raw.githubusercontent.com/jahnog/deeplearning-notes/master/Course2/images/randn.jpg).

When used for weight nitialization, `randn()` helps the weights to avoid being close to the extremes, allocating most of them in the centre of the range.

An intuitive way to see it is, for example, if you take the `sigmoid()` activation function.

You’ll remember that the slope near $0$ or near $1$ is extremely small, so the weights near those extremes will converge much more slowly to the solution, and having most of them near the centre will speed up convergence.

## Optimisation Algorithms

Suppose batch gradient descent is taking excessively long to find a vlaue of the parameters that achieves a small value for the cost function:

- Normalise the input data
- Try mini-batch gradient descent
- Try momentum, Adam, etc.
- Try better random initialisation
- Try tuning the learning rate $\alpha$

### Stochastic Gradient Descent

In Stochastic Gradient Descent, you use only 1 training example before updating the gradients. When the training set is large, SGD can be faster. But the parameters will "oscillate" toward the minimum rather than converge smoothly.

### Mini-Batch Gradient Descent

Two steps:

- Shuffle training set
- Partition

Typical mini-batch sizes: $2^6=64, 2^7=128, 2^8=256, 2^9=512; \gt 1, \lt m$

When the mini-batch size is the same as the training size $m$, mini-batch gradient descent is equivalent to batch gradient descent.

Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" toward convergence. Using momentum can reduce these oscillations.

**[Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/pdf/1711.00489.pdf)**

### Exponentially Weighted (Moving) Averages

$V_t$ can approximate the moving average over $\approx\frac{1}{1-\beta}$ time periods:

$$
V_t = \beta V_{t-1} + (1-\beta) \theta_t
$$

This is a key concept in a number of optimisation algorithms.

For the initial period, you can implement a *bias correction* while the algorithm is still warming up.

### Gradient Descent with Momentum

Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.

$$
V_{dW} = \beta V_{dW} + (1-\beta) dW
$$

$$
V_{db} = \beta V_{db} + (1-\beta) db
$$

$$
W = W - \alpha v_{dW}, b = b - \alpha v_{db}
$$

You think of $\beta$ as friction, $V_{dW}, V_{db}$ as velocities, and $dW, db$ as acceleration.

- Increasing $\beta$ will make the trend smoother and shift it slightly to the right.
- Decreasing $\beta$ will create more oscillation.

A common value of $\beta=0.9$, averaging over the last $1/1-\beta=10$ days.

This will almost always work better than normal gradient descent.

### RMSProp (Root Mean Squared Prop)

$$
S_{dW} = \beta S_{dW} + (1-\beta) dW^2
$$

### Adam (Adaptive Moment Adaptation)

Adam combines the advantages of RMSProp and Momentum. Its **advantages are that it requires little memory, and usually works well even with little tuning of hyperparameters except $\alpha$.**

1. It calculates an exponentially weighted average of past gradients, and stores it in variables $v$ (before bias correction) and $v^{corrected}$ (with bias correction).
2. It calculates an exponentially weighted average of the squares of the past gradients, and  stores it in variables $s$ (before bias correction) and $s^{corrected}$ (with bias correction).
3. It updates parameters in a direction based on combining information from "1" and "2".

The update rule is, for $l = 1, ..., L$:

$$\begin{cases}
v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1 - \beta_1) \frac{\partial \mathcal{J} }{ \partial W^{[l]} } \\
v^{corrected}_{dW^{[l]}} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t} \\
s_{dW^{[l]}} = \beta_2 s_{dW^{[l]}} + (1 - \beta_2) (\frac{\partial \mathcal{J} }{\partial W^{[l]} })^2 \\
s^{corrected}_{dW^{[l]}} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t} \\
W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}{\sqrt{s^{corrected}_{dW^{[l]}}} + \varepsilon}
\end{cases}$$
where:

- t counts the number of steps taken by Adam
- L is the number of layers
- $\beta_1$ and $\beta_2$ are hyperparameters that control the two exponentially weighted averages.
- $\alpha$ is the learning rate
- $\varepsilon$ is a very small number to avoid dividing by zero

### Learning Rate Decay

During the first part of training, your model can get away with taking large steps, but over time, using a fixed value for the learning rate alpha can cause your model to get stuck in a wide oscillation that never quite converges. But if you were to slowly reduce your learning rate alpha over time, you could then take smaller, slower steps that bring you closer to the minimum. This is the idea behind learning rate decay:

1. Decay on every iteration:
   $$\alpha = \frac{1}{1 + decayRate \times epochNumber} \alpha_{0}$$
2. Fixed interval scheduling (using floor division in the denominator):
   $$\alpha = \frac{1}{1 + decayRate \times \lfloor\frac{epochNum}{timeInterval}\rfloor} \alpha_{0}$$

With Mini-batch GD or Mini-batch GD with Momentum, the accuracy is significantly lower than with Adam, but when learning rate decay is added on top, either can achieve performance at a speed and accuracy score that's similar to Adam:

| Optimization method | Accuracy |
| --- | --- |
| Gradient descent | >94.6% |
| Momentum | >95.6% |
| Adam | 94% |

In the case of Adam, notice that the learning curve achieves a similar accuracy but faster.

### The Problem of Local Optima

In low-dimensional spaces, gradient descent may get stuck at local optima.

*In high-dimensional spaces*, most points where the gradient is zero are actually not local optima but saddle points of the cost functions.

It is therefore not very likely for gradient descent to get stuck at local optima in practice actually.

However, *plateaus may significantly slow down convergence.* Optimisation algorithms can speed up convergence.

## Hyperparameter Tuning

Most important:

- Learning rate $\alpha$

Quite important:

- Momentum term $\beta \approx 0.9$
- Number of hidden units
- (Mini-)batch size

Not that important:

- Number of layers
- Learning rate decay

Not important:

- Adam parameters use $\beta_1 \approx 0.9$ (i.e. sampling over last 10 data points), $\beta_2 \approx 0.999$ (i.e. last 1000 data points), $\epsilon \approx 10^{-8}$

*For **hyperparameter search**, sample using the log scale, not the linear scale!*

For example, to sample a random number between $10^{-3}$ and $10^0$ for the learning rate:

```python
r = -3 * np.random.rand()
alpha = 10**r
```

## Batch Normalisation: Normalising inputs z to speed up learning

Batch normalisation works because it helps reduce the internal covariance:

$$
z_{\text{norm}}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2} + \epsilon}
$$

The normalisation formula uses a smoothing parameter $\epsilon$ to prevent division by $0$ in case the variance is very small.

The parameters $\gamma^{[l]}$ and $\beta^{[l]}$ set the variance and mean of the linear variable $\tilde{z}^{[l]}$ of a given layer.

$$
\tilde{z}^{(l)} = \beta^{[l]} z_{\text{norm}}^{(l)} + \gamma^{[l]}
$$

It is okay to drop the parameter $b^{[l]}$ from the forward propagation because it will be subtracted out when computing $\tilde{z}^{[l]} = \gamma z_{\text{normalize}}^{[l]} + \beta^{[l]}$.

$\gamma$ and $\beta$ can be learned using Adam, Gradient Descent with Momentum, or RMSProp; although not with plain Gradient Descent.

After training a neural network with Batch Norm, at test time, to evaluate the neural network on a new example you should perform the needed normalisations, and use $\mu$ and $\sigma^2$ estimated using an exponentially weighted average across mini-batches seen during training.

## Multi-Class Classification: Softmax

Softmax is a generalisation of logistic regression to multiple classes:

$$
\hat{y} = a^{[L]} = g^{[L]}(z) = \frac{e^{z^{(i)}}}{\sum_{j=1}^n{e^{z^{(j)}}}}
$$

$$
L(\hat{y}, y) = -\sum_{j=1}^n{y_j \log(\hat{y}_j)}
$$

$$
J(x, b) = \frac{1}{m} \sum_{i=1}^m{L(\hat{y}^i, y^i)}
$$
