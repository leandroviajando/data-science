# [Machine Learning](https://aman.ai/coursera-ml/)

- [Loss Functions](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

## Linear Regression

$$
\hat{y}^{(i)} = f_{w,b}(x^{(i)}) = wx^{(i)} + b
$$

The squared error cost function is the most used with regression problems; it is a convex function that has a minimum:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m}(f_{w,b}(x^{(i)}) - y^{(i)})^2
$$

```python
def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        J (float): The cost of using w, b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]

    f = w @ x + b

    return np.sum(np.square(f - y)) / (2 * m)
```

## Gradient Descent

$$
\displaystyle \min_{w, b}{J(w, b)}
$$

$$
w := w - \alpha \frac{\partial}{\partial w}{J(w, b)}
$$

$$
b := b - \alpha \frac{\partial}{\partial b}{J(w, b)}
$$

```python
from typing import Tuple


def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float) -> Tuple[float, float]:
    """
    Computes the gradient for linear regression

    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model

    Returns
      dJ_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dJ_db (scalar): The gradient of the cost w.r.t. the parameter b
     """
    f = w @ x + b

    dJ_dw = np.mean((f - y) * X)
    dJ_db = np.mean(f - y)

    return dJ_dw, dJ_db
```

## Multiple Linear Regression

$$
f_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b
$$

Vectorisation is faster because it allows operations to be performed simultaneously on multiple elements of an array, utilizing the parallel processing capabilities of modern CPUs.

```python
f = w @ x + b
```

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)})^2
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
$$

**Feature scaling:** When you have different features that take on very different
ranges of values, it can cause gradient descent to run slowly. Normalising or standardising the different features so they all take on a comparable range of values
may speed up gradient descent significantly.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```

**Feature engineering:** Depending on what insights you
may have into the application,
rather than just taking
the features that you happen to have
started off with sometimes by defining new features,
you might be able to get a much better model.

**Polynomial regression** is a type of feature engineering that allows you to fit non-linear functions to features, e.g. $\text{house\_size}^3$ if a cubic shape is a better fit.

```python
x = np.arange(0, 20, 1)

X = np.c_[x, x**2, x**3]
```

## Logistic Regression

$$
g(z) = \frac{1}{1 + e^{-z}}, 0 < g(z) < 1, z = f_{\mathbf{w}, b}(\mathbf{x})
$$

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    return 1 / (1 + np.exp(-z))
```

$f_{\mathbf{w}, b}(\mathbf{x})$ may be the linear regression equation, or a polynomial regression, etc. This allows for learning different decision boundaries: lines, circles, etc.

The squared error loss used for linear regression does not work with logistic regression because the sigmoid causes it to have many local minima, i.e. it is not convex and gradient descent may fail to converge.

The binary cross entropy loss function (for a single training example), and the resulting cost function, derived through maximum likelihood estimation, are convex and can be used for binary classification:

$$
\begin{split}
L(f_{\mathbf{w}, b}(\mathbf{x^{(i)}}), y^{(i)})
 &= -y^{(i)} \log(f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) - (1 - y^{(i)}) \log(1 - f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) \\
 &= -1 \log(f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) \text{ if } y^{(i)} = 1 \\
 &= -(1-0) \log(1 - f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) \text{ if } y^{(i)} = 0 \\
\end{split}
$$

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m{L(f_{\mathbf{w}, b}(\mathbf{x^{(i)}}), y^{(i)})} = \frac{1}{m} \sum_{i=1}^m( -y^{(i)} \log(f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) - (1 - y^{(i)}) \log(1 - f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) )
$$

```python
def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, *argv) -> float:
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      J : (scalar) cost
    """
    y, w = y.reshape(-1, 1), w.reshape(-1, 1)

    f = sigmoid(X @ w + b)

    J = -y * np.log(f) - (1 - y) * np.log(1 - f)

    return np.mean(J)
```

Let $f_{\mathbf{w}, b}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$, then the gradient descent update steps can are the same as for linear regression:

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
$$

```python
from typing import Tuple


def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, *argv) -> Tuple[np.ndarray, float]:
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dJ_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
      dJ_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
    """
    f = sigmoid(X @ w + b)

    dJ_dw = np.mean((f - y) * X.T, axis=1)
    dJ_db = np.mean(f - y)

    return dJ_db, dJ_dw

def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    return np.round(sigmoid(X @ w.reshape(-1, 1) + b), 0).reshape(-1,)
```

## Overfitting (high variance)

To avoid overfitting:

- Collect more data
- Reduce number of features through feature selection
- Reduce the size of the parameters through regularization, which reduces a feature's importance instead of outright removing it

If the regularization parameter $\lambda$ is zero, the model will likely overfit (no regularization); if $\lambda$ is very large, all the weights go to zero and $f = b$, which is just a horizontal line (i.e. underfitting).

### Regularized Linear Regression

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}+ \frac{\lambda}{m} w_j
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
$$

Note $b$ isn't being regularized, so the update step for $b$ doesn't change.

### Regularized Logistic Regression

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m( -y^{(i)} \log(f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) - (1 - y^{(i)}) \log(1 - f_{\mathbf{w}, b}(\mathbf{x^{(i)}})) ) + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2
$$

```python
def compute_cost_reg(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1) -> float:
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      J : (scalar)     cost
    """

    m, n = X.shape

    cost_without_reg = compute_cost(X, y, w, b)

    reg_cost = lambda_ * np.sum(np.square(w)) / (2 * m)

    return cost_without_reg + reg_cost
```

Let $f_{\mathbf{w}, b}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$, then the gradient descent update steps can are the same as for linear regression:

$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}+ \frac{\lambda}{m} w_j
$$

$$
\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})
$$

Note $b$ isn't being regularized, so the update step for $b$ doesn't change.

```python
from typing import Tuple


def compute_gradient_reg(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1) -> Tuple[float, np.ndarray]:
    """
    Computes the gradient for logistic regression with regularization

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dJ_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
      dJ_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.
    """
    m, n = X.shape

    dJ_db, dJ_dw = compute_gradient(X, y, w, b)

    dJ_dw += lambda_ * w / m

    return dJ_db, dJ_dw
```

## Neural Networks

Linear Regression is a Single-Layer Neural Network with a linear activation function.

```python
import tensorflow as tf


linear_layer = tf.keras.layers.Dense(units=1, activation="linear")

linear_layer(X_train[0].reshape(1,1))

w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")
>> w = [[1.]], b=[0.]
```

Logistic Regression is a Single-Layer Neural Network with a sigmoid activation function.

Unlike regression, where some feature selection, engineering (polynomial features) may be necessary, neural networks are capable of learning features. For example, in a convolutional neural network, the first layer may learn to detect edges in the first layer, and so on.

Normalisation can also be implemented as a layer.

```python
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learn mean, variance
Xn = norm_l(X)
```

### Forward Propagation: Inference

```python
def my_dense(a_in: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes dense layer

    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units

    Returns
      a_out (ndarray (j,))  : j units|
    """
    a_out = g(a_in @ W + b)

    return a_out

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2

def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)
```

### Activation Functions

Why do we need activation functions?

A linear activation function is essentially equivalent to no activation function at all; it's just a linear regression, no matter how many layers the network has, because a linear function of a linear function is always a linear function. So in the hidden layers, a linear activation doesn't make much sense.

#### Hidden layers

The ReLU activation is often preferable to a sigmoid activation function because it does not suffer from vanishing gradients.

#### Output layer

- Binary classification -> sigmoid activation
- Regression -> linear or ReLU activation

### Softmax: Multiclass Classification

The softmax activation function and (sparse categorical, i.e. each digit can be `0` or `1`) crossentropy loss are a generalization of logistic regression and binary crossentropy loss:

$$
\begin{split}
a_1 &= \frac{e^{z_1}}{e^{z_1} + e^{z_1} + \dots + e^{z_N}} = P(y = 1 | \mathbf{x}) \\
 & \vdots \\
a_N &= \frac{e^{z_N}}{e^{z_1} + e^{z_1} + \dots + e^{z_N}} = P(y = 1 | \mathbf{x}) \\
a_j &= \frac{e^{z_j}}{\sum_{k=0}^{N-1}{e^{z_k}}}
\end{split}
$$

```python
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax converts a vector of values to a probability distribution.

    Args:
      z (ndarray (N,))  : input data, N features

    Returns:
      a (ndarray (N,))  : softmax of z
    """
    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z)
```

$$
L(a_1, ..., a_N, y) =
\begin{cases}
- \log(a_1) & \text{ if } y = 1 \\
- \log(a_2) & \text{ if } y = 2 \\
& \vdots \\
- \log(a_N) & \text{ if } y = N \\
\end{cases}
$$

Due to numerical round-off errors, it is recommended to use the `linear` output layer with parameter `from_logits=True` in the loss, instead of what would seemt to be the correct implementation:

```python
# may be mathematically unstable:
model = tf.keras.models.Sequential([
    # tf.keras.layers.InputLayer((400,)),
    tf.keras.layers.Dense(units=25, activation="relu"),
    tf.keras.layers.Dense(units=15, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax"),
])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy())

# better:
model = tf.keras.models.Sequential([
    # tf.keras.layers.InputLayer((400,)),
    tf.keras.layers.Dense(units=25, activation="relu"),
    tf.keras.layers.Dense(units=15, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="linear"),
])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
```

### Optimization algorithms (for Back Propagation)

**Adam (Adaptive Moment Estimation)** automatically adjusts the learning rate, if it makes sense to increase or decrease it, and usually is much faster than gradient descent.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
```

## Bias and variance

To reduce bias (high training error, compared to the baseline level of performance):

- Collect additional features or add polynomial features
- Decrease the regularization parameter $\lambda$

To reduce high variance (low training error, high validation error):

- Collect more training data
- Increase the regularization parameter $\lambda$

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=120, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(units=40, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    tf.keras.layers.Dense(units=6, activation="linear"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
```

## Skewed datasets

High **precision** means that if a diagnosis of a patient says that they have that rare disease, probably the patient does have it and it's an accurate diagnosis.

High **recall** means that if there's a patient with that a rare disease, probably the algorithm will correctly identify that they do have that disease.

In practice there's often a trade-off between precision and recall.

The most common way of combining precision recall is the **F1 score**, a way of combining the two metrics that gives more emphasis to whichever of these values is lower, i.e. the **harmonic mean.** Because it turns out if an algorithm has very low precision or very low recall is pretty not that useful.

## Decision Trees

**When to use decision trees?**

Decision trees work well with tabular (structured) data.

Neural networks work better with unstructured data (images, audio, text), and also work well with tabular (structured) data.

Benefits of using decision trees are that they are faster to train than neural networks, and relatively human-interpretable.

**Decision 1: How to choose what feature to split on at each node?**

Maximise purity (or minimise purity), e.g. left split: 10 of 10 emails are spam, right split: 0 of 10 emails are spam.

**Decision 2: When do you stop splitting?**

- When a node is 100% one class.
- When splitting a node will result in the tree exceeding a maximum depth.
- When improvements in purity are below a threshold.
- When number of examples in a node is below a threshold.

### Entropy

Entropy is a measure of impurity.

It peaks when there are equal numbers of both (/all) classes, and is zero when all instances belong to one class.

With $\log$ base $2$, entropy peaks at $1$:

$$
\begin{split}
H(p_1) &= -p_1 \log_2(p_1) - p_0 \log_2(p_0) \\
 &= -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1) \\
\end{split}
$$

For other bases of the $\log$, the curve of the function is scaled vertically.

```python
def compute_entropy(y: np.ndarray) -> float:
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    if len(y) == 0:
        return 0.0

    p1 = len(y[y == 1]) / len(y)
    if p1 == 0 or p1 == 1:
        return 0.0

    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
```

### Information Gain

A reduction of entropy is called information gain. This is the measure for choosing what features to use to split on at each node in a decision tree.

Information gain is calculated as the difference between the root node's entropy and the weighted average of its child branches' entropies:

$$
\text{Information Gain} = H(p_1^{root}) - \bigl( w^{left} H(p_1^{left}) + w^{right} H(p_1^{right}) \bigr)
$$

```python
def compute_information_gain(X: np.ndarray, y: np.ndarray, node_indices: np.ndarray) -> float:
    """
    Compute the information of splitting the node on a given feature

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e., the samples being considered in this step.

    Returns:
        information_gain (float):        Information gain computed

    """
    left_indices, right_indices = split_dataset(X, node_indices)

    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    information_gain = 0

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    weighted_entropy = (len(X_left) / len(X_node)) * left_entropy + (len(X_right) / len(X_node)) * right_entropy

    information_gain = node_entropy - weighted_entropy

    return information_gain
```

### Feature Engineering Decision Trees

- One-hot encode categorical features
- Split continuous features

Decision trees can also be used for *regression:* Choose the $N-1$ mid-points between the $N$ examples as possible splits, and find the split that gives the highest information gain. The prediction value is the weighted average of the instances on that sub-branch.

### Tree Ensembles

Trees are very sensitive to small changes in the data, and therefore not so robust.

Having lots of decision trees and having them vote, it makes your overall algorithm less sensitive to what any single tree may be doing because it gets only one vote out of three or one vote out of many, many different votes and it makes your overall algorithm more robust.

**Sampling with replacement** is used to create training sets for a bagged decision tree.

**Random forests** *additionally randomize feature choice:* At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k < n$ features and allow the algorithm to only choose from that subset of features. A common choice for $k$ is

$$
k = \sqrt{n}
$$

Instead of picking with equal probability $1/m$ (i.e. uncorrelated trees / random forests), **boosted trees** pick misclassified examples from previously trained trees more often with the objective of systematically minimising error. An example is `XGBoost` (eXtreme Gradient Boosting) which is an open-source implementation with built-in regularization.

## Clustering

A clustering algorithm looks at a number of data points and automatically finds data points that are related or similar to each other. Let's take a look at what that means.

### kMeans

Randomly initialise $K$ cluster centroids.

- Step 1: Assign each point to its closest centroid:

$$
J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_k) = \frac{1}{m} \sum_{i=1}^m{\lVert X^{(i)} - \mu_k \rVert^2}
$$

$$
\min_{c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_k}{J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_k)}
$$

```python
def find_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """
    m = X.shape[0]
    K = centroids.shape[0]

    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)  # broadcost norm of (m, K, n) - (K, n) -> (m, K)

    idx = np.argmin(distances, axis=1)  # (m,)

    return idx
```

- Step 2: Recompute the centroids to be the average (mean) of points assigned to cluster $k$

```python
def compute_centroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    centroids = np.array([
        np.mean(X[idx == k], axis=0) for k in range(K)
    ])

    return centroids
```

**Initialisation:** Different initialisations result in different clusters. Therefore, randomly initialise multiple times (e.g. 100), and the pick the set of clusters that gave the lowest cost $J$.

**Choosing the number of clusters:** Oftentimes, clustering is performed for some downstream use case. This is a good metric for choosing the best number of clusters. (Note the cost is always minimised for $K = N$).

## Anomaly Detection

Anomaly detection algorithms look at an unlabeled dataset of normal events and thereby learns to detect or to raise a red flag for if there is an unusual or an anomalous event.

The most common way to carry out anomaly detection, for example for fraud detection, is through a technique called density estimation:

- Choose $n$ features and model $p(x)$ from the data, where features $x_i$ are considered to be independent:

$$
p(\mathbf{x}) = p(x_1; \mu_1, \sigma_1^2) \times p(x_2; \mu_2, \sigma_2^2) \times \dots \times p(x_n; \mu_n, \sigma_n^2)
$$

- Fit parameters $\mu_1, \dots, \mu_n, \sigma_1^2, \dots, \sigma_n^2$ to find the maximum likelihood estimates of the mean and variance:

$$
\mu = \frac{1}{m} \sum_{i=1}^m{X^{(i)}}, \sigma^2 = \frac{1}{m} \sum_{i=1}^m{(X^{(i)} - \mu)^2}
$$

```python
def estimate_gaussian(X):
    """
    Calculates mean and variance of all features
    in the dataset

    Args:
        X (ndarray): (m, n) Data matrix

    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    mu = X.mean(axis=0)
    var = np.mean(np.square(X - mu), axis=0)

    return mu, var
```

- Identify unusual data points by checking which have $p(x) < \epsilon$:

$$
p(x) = \prod_{j=1}^n{p(x_j; \mu_j, \sigma_j^2)} = \prod_{j=1}^n{\frac{1}{\sqrt{2 \pi \sigma_j}} e^{- \frac{(x_j - \mu_j)^2}{2 \sigma_j^2}}}
$$

```python
def select_threshold(y_val, p_val):
    """
    Finds the best threshold to use for selecting outliers
    based on the results from a validation set (p_val)
    and the ground truth (y_val)

    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set

    Returns:
        epsilon (float): Threshold chosen
        F1 (float):      F1 score by choosing epsilon as threshold
    """
    best_epsilon = 0
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        anomalies = (p_val < epsilon)

        tp = np.sum((anomalies == 1) & (y_val == 1))
        fp = np.sum((anomalies == 1) & (y_val == 0))
        fn = np.sum((anomalies == 0) & (y_val == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
```

| Anomaly detection | Supervised learning |
| --- | --- |
| Very small number of positive examples | Large number of positive and negative examples |
| Many different "types" of anomalies; different for the algorithm to learn from positive examples what the anomalies look like | Enough positive examples for the algorithm to get a sense of what positive examples look like |
| Future anomalies may look nothing like any of the anomalous examples seen so far (e.g. new types of fraud) | Future positive examples likely to be similar to the ones in the training set (e.g. spam emails) |

**Feature selection** is even more important with anomaly detection because it will be difficult to ignore the given features, given the small number of positive examples.

*Normalise non-Gaussian features*, e.g. $\log$-transform

## Recommender Systems

- Collaborative filtering: Recommend items based on ratings of users who gave similar ratings
- Content-based filtering: Recommend items based on features of user and item features to find a good match

### Collaborative Filtering

$$
r(i, j) =
\begin{cases}
 1 & \text{if user j has rated movie} \\
 0 & \text{otherwise}
\end{cases}
$$

$m^{(j)}$ is the number of movies rated by user $j$.

For movie $i$ and user $j$:

$$
J(w^{(j)}, b^{(j)}) = \frac{1}{2} \sum_{i: r(i, j) = 1}{(w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2} + \frac{\lambda}{2} \sum_{k=1}^n{(w_k^{(j)})^2}
$$

For all users:

$$
J(\mathbf{w}, \mathbf{b}) = \frac{1}{2} \sum_{j=1}^{n_u}{\sum_{i: r(i, j) = 1}{(w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2}} + \frac{\lambda}{2} \sum_{j=1}^{n_u}{\sum_{k=1}^n{(w_k^{(j)})^2}}
$$

To learn ratings $\mathbf{x}$:

$$
J(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^{n_m}{\sum_{j: r(i, j) = 1}{(w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2}} + \frac{\lambda}{2} \sum_{i=1}^{n_m}{\sum_{k=1}^n{(x_k^{(i)})^2}}
$$

Combining both:

$$
\begin{split}
J(\mathbf{w}, \mathbf{b}, \mathbf{x}) &= \frac{1}{2} \sum_{(i, j): r(i, j) = 1}{(w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2} + \frac{\lambda}{2} \sum_{j=1}^{n_u}{\sum_{k=1}^n{(w_k^{(j)})^2}} + \frac{\lambda}{2} \sum_{i=1}^{n_m}{\sum_{k=1}^n{(x_k^{(i)})^2}} \\
 &= \frac{1}{2} \sum_{j=1}^{n_u}{\sum_{i=1}^{n_m}{r(i, j) \times(w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i, j)})^2}} + \frac{\lambda}{2} \sum_{j=1}^{n_u}{\sum_{k=1}^n{(w_k^{(j)})^2}} + \frac{\lambda}{2} \sum_{i=1}^{n_m}{\sum_{k=1}^n{(x_k^{(i)})^2}}
\end{split}
$$

```python
def cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering

    Args:
      X (ndarray (num_movies, num_features)): matrix of item features
      W (ndarray (num_users, num_features)) : matrix of user parameters
      b (ndarray (1, num_users)             : vector of user parameters
      Y (ndarray (num_movies, num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies, num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter

    Returns:
      J (float) : Cost
    """
    J = (1/2) * np.sum(R * ((X @ W.T + b - Y)**2)) \
        + (lambda_/2) * np.sum(W**2) \
        + (lambda_/2) * np.sum(X**2)

    return J
```

Optimization objective:

$$
\min_{\mathbf{w}, \mathbf{w}, \mathbf{b}}{J(\mathbf{w}, \mathbf{b}, \mathbf{x})}
$$

Gradient descent:

$$
w_i^{(j)} = w_i^{(j)} - \alpha \frac{\partial}{\partial w_i^{(j)}} J(\mathbf{w}, \mathbf{b}, \mathbf{x})
$$

$$
b^{(j)} = b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}} J(\mathbf{w}, \mathbf{b}, \mathbf{x})
$$

$$
x_k^{(i)} = x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}} J(\mathbf{w}, \mathbf{b}, \mathbf{x})
$$

With collaborative filtering, $x$ is also a parameter.

#### Binary labels

Many important applications of recommended systems or collective filtering algorithms involved binary labels where instead of a user giving you a one to five star or zero to five star rating, they just somehow give you a sense of they like this item or they did not like this item.

$$
J(\mathbf{w}, \mathbf{b}, \mathbf{x}) = \frac{1}{m} \sum_{(i, j): r(i, j) = 1}( -y^{(i, j)} \log(f_{w, b, x}(x)) - (1 - y^{(i, j)}) \log(1 - f_{w, b, x}(x)) )
$$

where $f_{w, b, x}(x) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})$

#### Mean Normalisation

Back in the first course, you have seen how for linear regression, feature normalization can help the algorithm run faster. In the case of building a recommender system with numbers such as movie ratings from one to five or zero to five stars, it turns out your algorithm *will run faster and also perform a bit better* if you first carry out mean normalization. That is, if you *normalize the movie ratings to have a consistent zero average value on a per-row basis:*

$$
y_{norm}(i, j) = y(i, j) - \mu_i \text{ where } \mu_i = \frac{1}{\sum_j{r(i, j)}} \sum_{j: r(i, j) = 1}{y(i, j)}
$$

### Content-Based Filtering

```python
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_outputs),
])

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_outputs),
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())
```

The neural network above produces two feature vectors, a user feature vector $v_u$, and a movie feature vector, $v_m$. These are 32 entry vectors whose values are difficult to interpret. However, similar items will have similar vectors. This information can be used to make recommendations. For example, if a user has rated "Toy Story 3" highly, one could recommend similar movies by selecting movies with similar movie feature vectors.

A similarity measure is the squared distance between the two vectors $\mathbf{v_m^{(k)}}$ and $\mathbf{v_m^{(i)}}$:
$$\left\Vert \mathbf{v_m^{(k)}} - \mathbf{v_m^{(i)}}  \right\Vert^2 = \sum_{l=1}^{n}(v_{m_l}^{(k)} - v_{m_l}^{(i)})^2\tag{1}$$

```python
def sq_dist(a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the squared distance between two vectors

    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features

    Returns:
      d (float) : distance
    """
    d = np.sum((a-b)**2)

    return d
```

## Reinforcement Learning

The goal of reinforcement learning is to choose a policy $\pi(s) = a$ that will tell us what action $a$ to take in state $s$ so as to maximise the expected return.

The **Bellman equation** is the return if you start from state $s$, take action $a$ (once), and then behave optimally after that:

$$
\begin{split}
Q^*(s, a)
&= R(s) + \gamma E[\max_{a'}{Q^*(s', a')}] \\
&= R_1 + \gamma E[R_2 + \gamma R_3 + \gamma^2 R_4 + \dots]
\end{split}
$$
