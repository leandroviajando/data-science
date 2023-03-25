# [Natural Language Processing with Sequence Models](https://www.coursera.org/learn/sequence-models-in-nlp?specialization=natural-language-processing)

## 3.1. Neural Networks for Sentiment Analysis

Previously in the course you did sentiment analysis with logistic regression and naive Bayes. Those models were in a sense more naive, and are not able to catch the sentiment off a tweet like: "I am not happy " or "If only it was a good day".

### [Trax](https://trax-ml.readthedocs.io/en/latest/)

Advantages of using a framework such as [JAX](https://jax.readthedocs.io/en/latest/index.html), PyTorch, TensorFlow:

- Runs fast on CPUs, GPUs, TPUs
- Parallel computing
- Records algebraic computations for gradient evaluation

### Dense Layer

$$
z^{[i]} = W^{[i]} a^{[i-1]}
$$

where the weights are the trainable parameter

### ReLU Layer

The ReLU layer keeps the network stable.

It is an activation layer that typically follows a dense fully connected layer, and transforms any negative values to 0 before sending them on to the next layer.

$$
g(z^{[i]}) = \max(0, z^{[i]})
$$

### Serial Layer

A serial layer is a composition of sublayers, e.g. dense and activation layers, in *serial* arrangement.

Calling a serial layer object will compute the forward propagation of the entire model.

```python
from trax import layers as tl

model = tl.Serial(
    tl.Dense(4),
    tl.Sigmoid(),
    tl.Dense(4),
    tl.Sigmoid(),
    tl.Dense(3),
    tl.Softmax(),
)
```

### Embedding Layer

Embeddings are trainable using an embedding layer by learning a matrix of weights of **size equal to the vocabulary** $V$ **times the dimension of the embedding** $N$.

### Mean Layer

The mean layer computes the **mean of the word embeddings** (see course 2).

It has no trainable parameters.

### Computing gradients with Trax

$$
f(x) = 3x^2 + x \rArr \frac{\partial f(x)}{\partial x} = 6x + 1
$$

```python
def f(x):
    return 3 * x**2 + x

grad_f = trax.math.grad(f)
```

### Training

The `grad()` function makes training much easier:

```python
y = model(x)

grads = grad(y.forward)(y.weights, x)

for ...:
    weights -= alpha * grads
```

## 3.2. Recurrent Neural Networks for Language Modelling

One of the biggest problems with N-gram modelling is that you end up using a lot of RAM and memory.

RNNs help mitigate this problem and outperform N-grams in language generation tasks.

Furthermore, you can have a bidirectional RNN that could keep track of information from both directions and allow for better character generation.

You can also add several layers to your RNN and make it deeper. This allows you to capture more abstract dependencies.

Finally, you will also learn about gated recurrent units, which are more powerful than vanilla RNNs.

### Traditional Language Models

**N-grams** are capable of capturing dependencies between distant words - at the expense of very high memory consumption.

$$
P(w_1, w_2, w_3) = P(w_1) \times P(w_2 | w_1) \times P(w_3 | w_2)
$$

### Recurrent Neural Networks

RNNs address the memory / RAM issues encountered with traditional language models.

They model relationships among distant words, and *a lot of these computations share parameters.*

*The number of parameters in an RNN is the same regardless of the input's length.*

### Applications of RNNs

- One-to-one
- One-to-many, e.g. caption generation
- Many-to-one, e.g. sentiment analysis, topic categorisation
- Many-to-many, e.g. machine translation

### Hidden states

- propagate information through time:

$$
\begin{split}
h^{\lt t \gt}
& = g(W_h[h^{\lt t-1 \gt}, x^{\lt t \gt}] + b_h) \\
& = g(W_{h,h} h^{\lt t-1 \gt} \oplus W_{h,x} x^{\lt t \gt} + b_h) \\
\end{split}
$$

where $\oplus$ means concatentation of the two matrices.

In Tensorflow, the hidden state corresponds to the `cur_value` in the `scan()` function:

```python
def scan(fn, elems, initializer=None, ...):
    cur_value = initializer
    ys = []

    for x in elems:
        y, cur_value = fn(x, cur_value)
        ys.append(y)

    return ys, cur_value
```

- and finally make predictions:

$$
\hat{y}^{\lt t \gt} = g(W_{y,h} h^{\lt t \gt} + b_y)
$$

### RNN Cost Function

Looking at a single example $(x, y)$,

$$
J = - \sum_{j=1}^K{y_j \log{\hat{y}_j}}
$$

where $y_j$ is either $0$ or $1$.

For RNNs, the Cross-Entropy Loss function is just an average through time:

$$
J = - \frac{1}{T} \sum_{t=1}^T \sum_{j=1}^K{y_j^{\le t \ge} \log{\hat{y}_j^{\le t \ge}}}
$$

Note there is a division by the number of time steps but not one for the number of classification categories - because there is just one non-zero value in every vector $y^{\leq t \geq}$.

### GRUs

GRUs tackle RNN's problem with loss of relevant information for long sequences of words.

**Relevance and update gates** to remember important prior information, e.g. first person, third person, singular, plural, etc.

Thus, GRUs "decide" how to update the hidden state, helping to preserve important information.

### Deep and Bi-Directional RNNs

In bi-directional RNNs, the outputs take information from the past and the future. They are acyclic graphs, which means that *the computations in one direction are independent from the ones in the other direction.*

Deep RNNs have more than one layer, which helps with complex tasks.

## 3.3. LSTMs and Named Entity Recognition

### RNNs

Advantages:

- Capture dependencies within a short range
- Take up less RAM than other N-gram models

Disadvantages:

- **Struggle to capture long-term dependencies**
- **Prone to vanishing or exploding gradients**

Backpropagation through time: The gradient is proportional to a sum of partial derivative products:

$$
\frac{\partial L}{\partial W_h} \propto \sum_{1 \leq k \leq t} \Biggl( \prod_{t \geq i \geq k} \frac{\partial h_k}{\partial h_{i-1}} \Biggr) \frac{\partial h_k}{\partial W_h}
$$

- Partial derivatives < 1 --> contribution goes to 0 --> vanishing gradient
- Partial derivatives > 1 --> contribution goes to infinity --> exploding gradient

Solving vanishing or exploding gradients:

- Identity RNN with ReLU activation
- Gradient clipping
- Skip connections, i.e. direct connections from earlier layers

### LSTMs

LSTMs offer a solution to vanishing gradients: *Learn when to remember and when to forget*

Typically, an LSTM will consist of:

- Three gates:
  1. Forget Gate: decides what information to keep

    $$f = \sigma(W_f[h_{t-1}; x_t] + b_f)$$

  2. Input Gate: decides what information to add

    $$i = \sigma(W_i[h_{t-1}; x_t] + b_i)$$

  3. Output Gate: decides what the next hidden state will be

    $$o = \sigma(W_o[h_{t-1}; x_t] + b_o)$$

- A cell state
  - Candidate cell state: information form the previous hidden state and current input
    - $\tanh$ ensures numerical stability of network by shrinking all values to be between -1 and 1 - thus preventing any of the values from the current inputs from becoming so large that they would make the other values insignificant

    $$g = \tanh(W_g[h_{t-1}; x_t] + g_g)$$

  - New cell state: add information from the candidate cell state using the forget and input gates

    $$c_t = f \odot c_{t-1} + i \odot g$$

- A hidden state
  - New hidden state: select information from the new cell state using the output gate (the tanh activation could be omitted)

    $$h_t = o_t \odot \tanh(c_t)$$

The gates allow the gradients to avoid vanishing and exploding

#### Applications

- Next-character prediction
- Chatbots
- Music composition
- Image captioning
- Speech recognition

### Named Entity Recognition (NER)

Locate and extract predefined entities (places, organisations, names, times and dates) from text

#### Data Processing

1. Assign each class a number
2. Assign each word a number
3. Token padding
   - For LSTMs, all sequences need to be the same size because, in a vectorised representation, equal sequence length allows for more efficient batch processing.
4. Create a data generator

#### Training a NER LSTM

1. Convert words and entities into same-length numerical arrays
2. Train in batches for faster processing
3. Run the output through a final layer and activation

```python
model = tl.Serial(
    tl.Embedding(),
    tl.LSTM(),
    tl.Dense(),
    tl.LogSoftmax(),
)
```

#### Evaluating the Model

- Pass the test set through the model
- Get arg max across the prediction array
- If padding tokens, remember to mask them when computing accuracy! They are not part of the data
- Compare with the true labels

```python
def evaluate_model(test_sentences, test_labels, model):
    pred = model(test_sentences)
    outputs = np.argmax(pred, axis=2)
    mask = ...
    accuracy = np.sum(outputs == test_labels) / float(np.sum(mask))

    return accuracy
```

## 3.4. Siamese Networks

Siamese networks quantify the difference or similarity between two instances.

`How old are you?` = `What is your age?`

`Where are you from?` $\neq$ `Where are you going?`

### Architecture

1. Pass two questions through the same Embedding-LSTM model architecture, outputting two vectors
   - You need to train **only one** set of weights!
2. Calculate the **cosine similarity** $-1 \leq \hat{y} \leq 1$
3. Classify
   - different if $\hat{y} \leq \tau$
   - same if $\hat{y} \gt \tau$

### Cost Function

Given an anchor, a positive and a negative statement (relative to the anchor),

$$
\text{diff} = s(A, N) - s(A, P)
$$

$$
s(v_1, v_2) = \cos(v_1, v_2) = \frac{v_1 \cdot v_2}{\lVert v_1 \rVert \lVert v_2 \rVert}
$$

### Triplets

This gives the **triplet loss**:

$$
L(A, P, N) = \max(\text{diff} + \alpha, 0)
$$

**Triplet Cost**:

$$
J = \sum_{i=1}^m{L(A^{(i)}, P^{(i)}, N^{(i)})}
$$

### One Shot Learning

No retraining required for new examples.
