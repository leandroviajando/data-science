# Neural Networks and Deep Learning

- [Forward and Backward Propagation](https://jonaslalin.com/2021/12/10/feedforward-neural-networks-part-1/)
- [Activation Functions](https://jonaslalin.com/2021/12/21/feedforward-neural-networks-part-2/)
- [Cost Functions](https://jonaslalin.com/2021/12/22/feedforward-neural-networks-part-3/)

A neuron computes a linear function $z = Wx + b$ followed by an activation function.

## (Shallow) Neural Networks

Logistic regression is a "shallow" neural network. It has only one layer.

### Activation Functions

- **Sigmoid:** use as output layer in binary classification

$$
a = g(z) = \frac{1}{1 + e^{-z}}, a \in (0, 1)
$$

$$
\frac{d}{dz}g(z) = \frac{1}{1 + e^{-z}} (1 - \frac{1}{1 + e^{-z}}) = g(z) (1 - g(z))
$$

- **tanh:** similar and usually superior to sigmoid as a hidden layer

$$
a = g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, a \in (-1, 1)
$$

$$
\frac{d}{dz}g(z) = 1 - (\tanh(z))^2 = 1 - (g(z))^2
$$

- **ReLU:** better-defined gradient than sigmoid and tanh, and neural network will learn much faster

$$
a = g(z) = max(0, z), a \in [0, \infty)
$$

$$
\frac{d}{dz} g(z) =
\begin{cases}
  0 & \text{if } z \lt 0 \\
  1 & \text{if } z \geq 0
\end{cases}
$$

- **Leaky ReLU**

$$
a = g(z) = max(0.01z, z)
$$

$$
\frac{d}{dz} g(z) =
\begin{cases}
  0.01 & \text{if } z \lt 0 \\
  1 & \text{if } z \geq 0
\end{cases}
$$

Note a linear hidden layer is just normal linear regression, and a linear hidden layer with a sigmoid output layer is just logistic regression. Hence the need for non-linear hidden-layer activation functions to learn more complex structures; and then using a linear output layer for regression, and a sigmoid output layer for binary classification.

### Random Initialization

If you initialise weights to all zeros, all neurons / hidden layers will be identical. So this doesn't make sense because, no matter how many layers you have, all units in the network will be symmetric and compute exactly the same function.

*The bias term can be initialised as zero but the weights need to be initialised to small random numbers.*

```python
W = np.random.randn((2, 2)) * .01
```

## Deep Neural Networks

The initial layers detect "simpler" features, and the deeper layers detect more "complex" features.

**Forward propagation:**

$$
\begin{align}
Z^{[1]} &= W^{[1]} \cdot X + b^{[1]} \\
A^{[1]} &= g^{[1]}(Z^{[1]}) \\
\vdots \\
Z^{[l]} &= W^{[l]} \cdot A^{[l-1]} + b^{[l]} \\
A^{[l]} &= g^{[l]}(Z^{[l]}) \\
\vdots \\
A^{[L]} &= g^{[L]}(Z^{[L]}) = \hat{Y}
\end{align}
$$

Z is cached during forward propagation as it contains useful values for backward computation to compute derivatives.

**Backward propagation:**

$$
\begin{align}
dZ^{[L]} &= A^{[L]} - Y \\
dW^{[L]} &= \frac{1}{m} dZ^{[L]} A^{[L-1]^T} \\
db^{[L]} &= \frac{1}{m} \text{np.sum}(dZ^{[L]} \text{, axis=1, keepdims=True}) \\
dZ^{[L-1]} &= W^{[L]^T} dZ^{[L]} \star g'^{[L-1]}(Z^{[L-1]}) \\
\vdots \\
dZ^{[1]} &= W^{[2]^T} dZ^{[2]} \star g'^{[1]}(Z^{[1]}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} X \\
db^{[1]} &= \frac{1}{m} \text{np.sum}(dZ^{[1]} \text{, axis=1, keepdims=True})
\end{align}
$$

**Parameters:** $w, b$

Assume we store the values for $n^{[l]}$ in an array: `layer_dims = [4, 3, 2, 1]`. So layer 1 has four hidden units, layer 2 has 3 hidden units and so on. How do you initialize the parameters for the model?

```python
for l in range(len(layer_dims)):
    parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
    parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
```

**Hyperparameters:** Learning rate $\alpha$, number of iterations, hidden layers, hidden units, activation functions, etc.
