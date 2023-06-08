# [Sequence Models](https://deeplearningmath.org/sequence-models.html)

## Recurrent Neural Networks

Examples of sequence data:

- Speech recognition
- Music generation
- Sentiment classification
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Named entity recognition

Why not a standard network?

- Inputs and outputs will have different lengths in different examples
- Features learned across different positions aren't shared

Suppose your training examples are sentences (sequences of words); then $x^{(k)\lt l \gt}$ refers to the $l^{\text{th}}$ word in the $k^{\text{th}}$ training example.

Different types of RNNs: e.g.

- Sentiment classification: many-to-one, one-to-one
- Music generation: one-to-many
- Machine translation: many-to-many (encoder-decoder)

$$
a^{<0>} = 0
$$

$$
a^{\lt t \gt} = g(W_{a_a} a^{\lt t-1 \gt} + W_{a_x} x^{\lt t \gt} + b_a) = g(W_a[a^{\lt t-1 \gt}, x^{\lt t \gt}] + b_a), \text{ where } g = \tanh \text{ or ReLU}
$$

$$
\hat{y}^{\lt t \gt} = g(W_{y_a} a^{\lt t \gt} + b_y) = g(W_y a^{\lt t \gt} + b_y), \text{ where } g = sigmoid
$$

$$
L(\hat{y}, y) = \sum_{t=1}^{T_y}{L^{\lt t \gt}(\hat{y}^{\lt t \gt}, y^{\lt t \gt})} = \sum_{t=1}^{T_y}{- y^{\lt t \gt} \log \hat{y}^{\lt t \gt} - (1 - y^{\lt t \gt}) \log(1 - \hat{y}^{\lt t \gt})}
$$

### Language Models and Sequence Generation

$$
L(\hat{y}, y) = \sum_{t=1}^{T_y}{L^{\lt t \gt}(\hat{y}^{\lt t \gt}, y^{\lt t \gt})} = - \sum_{t=1}^{T_y} \sum_i{y_i^{\lt t \gt} \log \hat{y}_i^{\lt t \gt}}
$$

$$
P(y^{(1)}, y^{(2)}, y^{(3)}) = P(y^{(1)}) P(y^{(2)} | y^{(1)}) P(y^{(3)} | y^{(1)}, y^{(2)})
$$

where at the $t^{\text{th}}$ time step the RNN is estimating $P(y^{\lt t \gt} | y^{\lt 1 \gt}, y^{\lt 2 \gt}, ..., y^{\lt t-1 \gt})$

### Sampling Novel Sequences

Use the probabilities output by the RNN to randomly sample a chosen word for that time step as $\hat{y}^{\lt t \gt}$. Then pass this selected word to the next time step.

### Gated Recurrent Units (GRU)

A memory cell $c$ will remember, for example, if the subject of the sentence ("the cat") was singular or plural, in order to know whether to use "was" or "were" later in the sentence.

$$
c^{\lt t \gt} = a^{\lt t \gt}
$$

At every time step, we're going to consider overwriting the memory cell:

$$
\tilde{c}^{\lt t \gt} = \tanh(\Gamma_r * W_c[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_c)
$$

The relevance gate makes the process more robust:

$$
\Gamma_r = \sigma(W_r[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_r)
$$

The update gate (which is  almost always either $0$ or $1$) decides whether or not to update the memory cell:

$$
\Gamma_u = \sigma(W_u[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_u)
$$

$$
c^{\lt t \gt} = \Gamma_u * \tilde{c}^{\lt t \gt} + (1-\Gamma_u) * c^{\lt t-1 \gt}
$$

### Exploding Gradients

If the weights and activations are all taking on the value `NaN`, then you have an exploding gradient problem.

Exploding gradients happen when large error gradients accumulate and result in very large updates to the NN model weights during training. These weights can become too large and can cause an overflow, identified as `NaN`.

*Gradient clipping* is a solution that can be applied.

### Vanishing Gradients

In order to simplify the GRU without vanishing gradient problems even when training on very long sequences you should remove $\Gamma_r$, i.e. set $\Gamma_r = 1$ always.

If $\Gamma_u \approx 0$ for a timestep, the gradient can propagate back through that timestep without much decay. For the signal to backpropagate without vanishing, we need $c^{\lt t \gt}$ to be highly dependent on $c^{\lt t-1 \gt}$.

### Long Short Term Memory (LSTM)

| GRU | LSTM |
| --- | ---- |
| $\tilde{c}^{\lt t \gt} = \tanh(\Gamma_r * W_c[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_c)$ | $\tilde{c}^{\lt t \gt} = \tanh(W_c[a^{\lt t-1 \gt}, x^{\lt t \gt}] + b_c)$ |
| $\Gamma_u = \sigma(W_u[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_u)$ | $\Gamma_u = \sigma(W_u[a^{\lt t-1 \gt}, x^{\lt t \gt}] + b_u)$ |
| $\Gamma_r = \sigma(W_r[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_r)$ | $\Gamma_f = \sigma(W_f[a^{\lt t-1 \gt}, x^{\lt t \gt}] + b_f)$ |
| | $\Gamma_o = \sigma(W_o[c^{\lt t-1 \gt}, x^{\lt t \gt}] + b_o)$ |
| $c^{\lt t \gt} = \Gamma_u *\tilde{c}^{\lt t \gt} + (1-\Gamma_u)* c^{\lt t-1 \gt}$ | $c^{\lt t \gt} = \Gamma_u *\tilde{c}^{\lt t \gt} + \Gamma_f* c^{\lt t-1 \gt}$ |
| $a^{\lt t \gt} = c^{\lt t \gt}$ | $a^{\lt t \gt} = \Gamma_o * c^{\lt t \gt}$ |

### Bi-Directional RNNs

Acyclic graph that allows network to take into account information from before and after.

$$
\hat{y}^{\lt t \gt} = g(W_y[a_{\rarr}^{\lt t \gt}, a_{\larr}^{\lt t \gt}] + b_y)
$$

The disadvantage is that you need the entire sequence before being able to make any predictions.

### Deep RNNs

## Natural Language Processing & Word Embeddings

### Introduction to Word Embeddings

#### Word Representation

One-hot encoding is a basic representation of a dictionary that we've seen so far.

The inner product between any two one-hot encoded vectors is always 0, so it is not possible to represent similarities between words.

#### Word Embeddings

Word embeddings are a featurised representation. Its dimension will usually range between $50$ and $1000$ to capture the words' features.

When training word embeddings, we pick a given word and try to predict its surrounding words or vice versa.

**t-SNE** is a nin-linear dimensionality reduction technique that can be used to represent word embeddings.

**Cosine similarity:** if u and v are very similar, their inner product will tend to be large:

$$
sim(u, v) = \frac{u^T v}{\lVert u \rVert \lVert v \rVert}
$$

For example, $e_{\text{man}} - e_{\text{woman}} \approx e_{\text{uncle}} - e_{\text{aunt}}$, $e_{\text{man}} - e_{\text{uncle}} \approx e_{\text{woman}} - e_{\text{aunt}}$.

*Word vectors empower your model with an incredible ability to generalize:* Even if a specific word does not appear in the training set, the embedding RNN might reasonably be expected to recognise it / its relative similarity.

#### Word2Vec

Estimate $P(t|c)$, where $t$ is the target word and $c$ is a context word - both chosen from the training set to be nearby words.

**Skip-gram model:** $e_c = E o_c$ followed by a softmax activation

$$
P(t|c) = \frac{e^{\theta_t^T e_C}}{\sum_{t'=1}^{10,000}{e^{\theta_t^T e_C}}}
$$

For example, in this 100-dimensional word embedding trained on a 10,000 word vocabulary, $\theta_t$ and $e_C$ are both 100-dimensional vectors, trained with an optimisation algorithm:

$$
\min \sum_{i=1}^{10,000} \sum_{j=1}^{10,000} f(X_{ij})(\theta_i^T e_j + b_i + b_j' - \log X_{ij})^2
$$

where

- $X_{ij}$ is the number of times word $j$ appears in the context of word $i$
- $\theta_i$ and $e_j$ should be initialised randomly at the beginning of training
- theoretically, the weighting function $f$ must satisfy $f(0) = 0$.

A hierarchical softmax classifier (a tree) has asymptotic run time $\log \lvert v \rvert$ (where $\lvert v \rvert$ is the size of the dictionary) and can thus speed up the computationally expensive softmax computation.

#### Negative Sampling

Given a positive example, sample a number of negative examples.

#### GloVe Word Vectors

The GloVe algorithm is not used as much as Word2Vec or skip-gram models, although enjoys some popularity due to its simplicity.

$X_{ij}$ is the number of times that $i$ ($t$) appears in context ($c$) of $j$

### Applications using Word Embeddings

#### Transfer learning

Having trained word embeddings using a text dataset of $s_1$​ words, it makes sense to use these word embeddings for a language task for which you have a separate labelled dataset of $s_2$ words, if $s_1 \gt\gt s_2$.

#### Sentiment Classification

#### Debiasing Word Embeddings

## Sequence Models & Attention Mechanism

### Sequence Models

An encoder-decoder model for machine translation models the probability of the output sentence $y$ conditioned on the input sentence $x$:

$$
\argmax_{y^{\lt 1 \gt}, ... , y^{\lt T_y \gt}}{P(y^{\lt 1 \gt}, ... , y^{\lt T_y \gt} | x)}
$$

Why not a greedy search? May result in sub-optimal results; similar to how dynamic programming is preferable to greedy algorithms in some cases.

#### Beam Search

The Beam Search algorithm has a parameter $B$ - the beam width, which shortlists the $B$ most likely next words.

$$
P(y^{\lt 1 \gt}, y^{\lt 2 \gt} | x) = P(y^{\lt 1 \gt} | x) P(y^{\lt 2 \gt} | x, y^{\lt 1 \gt})
$$

Unlike BFS or DFS, Beam Search runs faster but is not guaranteed to find the exact maximum for $\max_y{P(y|x)}$.

A large $B$ will give a better result at the cost of being slower and using more memory; while a smaller $B$ will run faster and need less memory although give a worse result.

#### Length normalisation

*In machine translation, if you carry out beam search without using sentence normalisation, the algorithm will tend to output overly short translations.*

By taking logs, you end up with a more numerically stable algorithm
that is less prone to numerical rounding errors or really numerical under-floor. Because the logarithmic function is a strictly monotonically increasing function, we know that maximizing log p of y given x should give you the same result as maximizing p of y given x as in the same value of y that maximizes, this should also maximize that.

$$
\argmax_y{\prod_{t=1}^{T_y}{P(y^{\lt t \gt} | x, y^{\lt 1 \gt}, ... , y^{\lt t-1 \gt})}}
$$

$$\dArr$$

$$
\argmax_y{\frac{1}{T_y^{\alpha}} \sum_{t=1}^{T_y}{\log P(y^{\lt t \gt} | x, y^{\lt 1 \gt}, ... , y^{\lt t-1 \gt})}}
$$

Normalising by the number of words in your translation takes the average of the log of the probability of each word and significantly reduces the penalty for longer translations.

$0 \lt \alpha \lt 1$ is a hyperparameter that can be tuned, e.g. $\alpha=.7$.

This is called the **normalised log likelihood objective.**

#### Error Analysis in Beam Search

Analyse examples to see whether the beam search algorithm or the RNN model are at fault for incorrect predictions (where there is another word that maximises the probability, rather than the one chosen):

- If you find that beam search is responsible for a lot of errors ($P(y^*|x) \geq P(\hat{y}|x)$), then maybe it's worth working hard to increase the beam width.
- Whereas in contrast, if you find that the RNN model is at fault ($P(y^*|x) \lt P(\hat{y}|x)$), then you could do a deeper layer of analysis to try to figure out if you want to add regularization, or get more training data, or try a different network architecture, or something else.

#### BLEU Score (Bilingual Evaluation Understudy)

One of the challenges of machine translation is that, given a French sentence, there could be multiple English translations that are equally good translations of that French sentence.

Given a machine generated translation, the BLEU score is an automatically computed score that measures how good that machine translation is. And so long as the machine generated translation is pretty close to any of the references provided by humans, then it will get a high BLEU score.

### Attention Mechanism

The attention mechanism addresses the problem of long sequences; it is an RNN sequence model with context vectors as input:

$$
c^{\lt t \gt} = \sum_{t'}{\alpha^{\lt t, t' \gt} a^{\lt t' \gt}}
$$

where $\alpha^{\lt t, t' \gt}$ is the amount of attention $y^{\lt t \gt}$ should pay to $a^{\lt t' \gt}$:

$$
\alpha^{\lt t, t' \gt} = \frac{\exp{e^{\lt t, t' \gt}}}{\sum_{t'=1}^{T_x}{\exp{e^{\lt t, t' \gt}}}}
$$

This is a softmax so it sums to 1, i.e. $\sum_{t'}{\alpha^{\lt t, t' \gt}} = 1$.

$e^{\lt t, t' \gt}$ can be trained as a neural network with $s^{\lt t-1 \gt}$ and $a^{\lt t' \gt}$ as inputs.

### Speech Recognition - Audio Data

#### Connectionist temporal classification (CTC) cost for speech recognition

For example, `kk_eee____ee_p__eeeeeee_____rrrrr = keeper`.

#### Trigger Word Detection

$x^{\lt t \gt}$ represents the features of the audio (such as the spectrogram features) at time $t$.

## Transformers

RNNs, GRUs and LSTMs are sequential models, increasing in complexity so as to deal with longer-term dependencies.

The transformer architecture combines an attention-based representation with CNN-style parallel computation.

Unlike its predecessors RNNs, GRUs and LSTMs, it has a parallel architecture, and can process entire sentences all at the same time.

### Self-Attention

For each word, calculate its attention-based vector representation: *Given a word, its neighboring words are used to compute its context by summing up the word values to map the Attention related to that given word.*

$$
A(q, K, V) = \sum_i{\frac{\exp(q * k^{\lt i \gt})}{\sum_j{\exp(q * k^{\lt j \gt})}} v^{\lt i \gt}}
$$

This is also called the **scaled dot product attention:**

$$
Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V
$$

The queries, keys and values $q^{\lt i \gt} = W^Q x^{\lt i \gt}, k^{\lt i \gt} = W^K x^{\lt i \gt}, v^{\lt i \gt} = W^V x^{\lt i \gt}$ are learned.

- The query lets you ask a question about that word, such as what's happening in "Africa".

`Q = interesting questions about the words in a sentence`

- The key looks at all of the other words, and by the similarity to the query, helps you figure out which words gives the most relevant answer to that question. In this case, "visite" is what's happening in "Africa", someone's visiting Africa.

`K = qualities of words given a Q`

- The value allows the representation to plug in how visite should be represented within $A^{\lt 3 \gt }$, within the representation of "Africa". This allows you to come up with a representation for the word "Africa" that says this is Africa and someone is visiting Africa.

`V = specific representations of words given a Q`

This is a much more nuanced, much richer representation for the world than if you just had to pull up the same fixed word embedding for every single word without being able to adapt it based on what words are to the left and to the right of that word.

### Multi-Head Attention

Ask multiple questions for every single word:

$$
\text{MultiHead}(Q, K, V) = \text{concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h) W_O
$$

where

$$
\text{head}_i = \text{Attention}(W_i^Q Q, W_i^K K, W_i^V V)
$$

### Transformer Networks

The transformer network differs from the attention model in that the transformer network also uses a positional encoding.

Positional encoding

- provides extra information to the model
- is important because position and word order are essential in sentence construction of any language
- uses a combination of sine and cosine equations

A good positional encoding algorithm

- should output a unique encoding for each time step (i.e. word's position in a sentence)
- the distance between any two time steps should be consistent for all sentence lengths
- the algorithm should be able to generalise to longer sentences

The Decoder's output layers are a linear layer followed by a softmax layer.
