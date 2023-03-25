# [Natural Language Processing with Probabilistic Models](https://www.coursera.org/learn/probabilistic-models-in-nlp?specialization=natural-language-processing)

## 2.1. Autocorrect

1. Identify a misspelled word

    ```python
    if word not in vocab:
        misspelled = True
    ```

2. Find strings `n` edit distance away - an *edit* is an operation changing the string: For example, the minimum edit distance between `deep` and `creepy` is 4.
   - Insert (1 operation)
   - Delete (1 operation)
   - Switch (1 operation; two adjacent characters, not any characters)
   - Replace (2 operations)
3. Filter candidates
4. Calculate word probabilities

$$P(w) = \frac{C(w)}{M}$$

where `C` is the number of occurrences of a word, and `M` is the total size of the corpus.

### Minimum Edit Distance

How to evaluate similarity between two strings? *The minimum number of edits needed to transform one string into the other.*

Applications: Spelling correction, document similarity, machine translation, DNA sequencing

### Minimum Edit Distance Algorithm

The **Levenshtein distance** specifies the cost per operation.

If you need to reconstruct the path of how you got from one string to the other, you can use **backtracking**: Keep a simple pointer in each cell letting you know where you came from to get there. So you know the path taken across the table from the top left corner, to the bottom right corner. You can then reconstruct it.

This method for computation instead of brute force is a technique known as **dynamic programming:** You solve the smallest subproblem first, and then reuse that result to solve the next biggest subproblem, saving that result, reusing it again, and so on.

## 2.2. Part of Speech (POS) Tagging and Hidden Markov Models

### Part of Speech (POS) Tags

| lexical term | tag | example |
| ------------ | --- | ------- |
| noun | NN | something, nothing |
| verb | VB | learn, study |
| determiner | DT | the, a |
| w-adverb | WRB | why, where |

For example, `Why not learn something?` = `WRB` + `RB` + `VB` + `NN`

Because POS tags describe the characteristic structure of lexical terms in a sentence or text, you can use them to make assumptions about semantics. They're used for

- **named entity recognition**: In a sentence such as
`the Eiffel Tower is located in Paris`, `Eiffel Tower` and `Paris` are both named entities.
- **coreference resolution**: If you have the two sentences, `the Eiffel Tower is located in Paris`, and `it is 324 meters high`, you can use part-of-speech tagging to infer that it refers in this context to the Eiffel Tower.
- **speech recognition**, where you use parts of speech tags to check if a sequence of words has a high probability or not.

### Markov Chains

POS tags can be modelled as states in a Markov chain.

A state (i.e. POS tag) refers to a certain condition of the present moment. Given a state, one can identify the **transition probability** of the next word being another state (or the same state).

**Markov property:** the probability of the next event only
depends on the current event. This helps keep the model simple.

The transition probabilities between the different states / POS tags can be represented in a **transition matrix.** The sum for all transition probabilities outgoing form a given state (i.e. a row in the matrix) should always sum to one.

### Hidden Markov Models

While transition probabilities identify the probability of transitioning from one POS to another, **emission probabilities** give the probability of going from one state (POS tag) to a specific word.

$$\text{States } Q = \{q_1, ..., q_N\}$$

$$\text{Transition matrix } A = \begin{pmatrix}
a_{1,1} & \cdots & a_{1,N} \\
\vdots & \ddots & \vdots \\
a_{N+1,1} & \cdots & a_{N+1,N}
\end{pmatrix}$$

$$\text{Emission matrix } A = \begin{pmatrix}
b_{1,1} & \cdots & b_{1,V} \\
\vdots & \ddots & \vdots \\
b_{N,1} & \cdots & b_{N,V}
\end{pmatrix}$$

where `V` is the size of the corpus; and , for both matrices, the sum of each row has to be `1`; i.e. $\sum_{j=1}^{N}{a_{i,j}} = 1$ and $\sum_{j=1}^{V}{b_{i,j}} = 1$

#### Calculating Transition Probabilities

Count the number of times tag $t_{i−1}$​ is followed by $t_{i}$, and divide by the total number of times $t_{i−1}$ shows up (i.e. the number of times it shows up followed by anything else):

$$P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_{i})}{\sum_{j=1}^{N}{C(t_{i-1}, t_{j})}}$$

#### Populating the Transition Matrix

When populating the transition matrix, use **smoothing** to avoid division by zero: Add a small value $\epsilon$ to every transition probability:

$$P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_{i}) + \epsilon}{\sum_{j=1}^{N}{C(t_{i-1}, t_{j}) + N \times \epsilon}}$$

- For the minority of cases, this allows us to increase the probabilities in the transition and emission matrices and this allows us to have non zero probabilities.
- For the majority of cases, this allows us to decrease the probabilities in the transition and emission matrices and this allows us to have non zero probabilities.

#### Populating the Emission Matrix

$$P(w_i | t_i) = \frac{C(t_i, w_i) + \epsilon}{\sum_{j=1}^{N}{C(t_i, w_j) + N \times \epsilon}}$$

where $C(t_i,w_i)$ is the count associated with how many times the tag $t_i$ is associated with the word $w_i$.

### Viterbi Algorithm

The Viterbi algorithm is a graph algorithm making use of dynamic programming. It consists of three steps:

1. Initialization step
2. Forward step
3. Backward pass

Implementation:

- In Python indices start with 0
- Use log probabilities

**Previously, we have been multiplying the raw probabilities, but in reality we take the log of those probabilities. Probabilities are bounded between 0 and 1, and as a result the numbers could be too small and will go towards 0.**

## 2.3. Autocomplete and Language Models

Applications of N-Gram models:

- Speech recognition
- Auto complete
- Auto correct
- Augmentative communication (e.g. Stephen Hawking)

### N-Grams and Probabilities

An N-Gram is a sequence of $N$ words. The order of the words matters.

Example corpus of size $m=7$: `I am happy because I am learning.`

- Probability of a unigram $P(w) = \frac{C(w)}{m}$, e.g. $P(I) = \frac{2}{7}$, $P(\text{happy}) = \frac{1}{7}$
- Probability of a bigram $P(y|x) = \frac{C(\text{x y})}{\sum_{w}{C(\text{x w})}} = \frac{C(\text{x y})}{C(x)}$, e.g. $P(\text{am} | \text{I}) = \frac{C(\text{I am})}{C(\text{I})} = \frac{2}{2} = 1$
- Probability of a trigram $P(w_3 | w_1^2) = \frac{C(w_1^2 w_3)}{C(w_1^2)}$, e.g. $P(\text{happy} | \text{I am}) = \frac{C(\text{I am happy})}{C(\text{I am})} = \frac{1}{2}$
- Probability of an N-Gram $P(w_N | w_1^{N-1}) = \frac{C(w_1^{N-1} w_N)}{C(W_1^{N-1})}$

### Sequence Probabilities

What is the probability of a sequence?

$$P(B|A) = \frac{P(A, B)}{P(A)} = \frac{P(A) P(B|A)}{P(A)}$$

This can be generalised to the **chain rule**:

$$P(A, B, C, D) = P(A) P(B|A) P(C|A, B) P(D|A,B,C)$$

For example,

$$P(\text{the teacher drinks tea}) = P(\text{the}) P(\text{teacher} | \text{the}) P(\text{drinks} | \text{the teacher}) P(\text{tea} | \text{the teacher drinks})$$

#### Shortcomings

Problem: The corpus will almost never contain the exact sentence we're interested in.

#### N-Gram Approximation

Just consider the word directly before the next word:

$$P(\text{tea} | \text{the teacher drinks}) \approx P(\text{tea} | \text{drinks})$$

**Markov assumption:** only last $N$ words matter

- Bigram: $P(w_n | w_1^{n-1}) \approx P(w_n | w_{n-1})$
- N-Gram: $P(w_n | w_1^{n-1}) \approx P(w_n | w_{n-N+1}^{n-1})$

### Starting and Ending Sentences

- Start of sentence symbol: `<s>`
- End of sentence symbol: `</s>`

### The N-Gram Language Model

#### 1. Count Matrix

Rows: unique corpus (N-1)-grams
Columns: unique corpus words

#### 2. Probability Matrix

Divide each cell by its row sum

#### 3. Language Model

This gives a language model for sentence probability and next word prediction.

#### Log probability to avoid underflow

All probabilities in calculatoin $<=1$, and multiplying them results in risk of underflow

Better to use log probabilities in probability matrix and calculation:

$$\log(P(w_1^n)) \approx \sum_{i=1}^n{\log(P(w_i | w_{i-1}))}$$

$$P(w_1^n) = \exp(\log(P(w_1^n)))$$

$$\log(a \times b) = \log(a) + \log(b)$$

#### Generative Language Model

1. Choose sentence start
2. Choose next bigram starting with previous word
3. Continue until `</s>` is picked

### Language Model Evaluation

You can think of perplexity as a measure of the complexity in a sample of texts. A text that is written by humans is more likely to have a lower perplexity score; words chosen at random will have a high perplexity score.

### Out of Vocabulary Words

In some tasks like speech recognition or question answering, you will encounter and generate words only from a fixed set of words. Hence, a closed vocabulary.

Open vocabulary means that you may encounter words from outside the vocabulary, like a name of a new city in the training set.

Use a special tag for such unkown words. But use it sparingly.

The more unknown tokens there are, the lower the *perplexity score.* The higher the perplexity score the more our corpus will make sense.

### Smoothing

Smoothing pretends that each word appears at least once. It can be implemented for previously unseen N-Grams.

- 1-smoothing
- k-smoothing
- back off

## 2.4. Word Embeddings

Applications:

- Semantic analogies and similarity
- Sentiment analysis
- Classification of customer feedback

Advanced applications:

- Machine translation
- Information extraction
- Question answering

### Basic word representations

- Integers: `a - 1, able - 2, about - 3, ...`
  - Simple but the ordering makes little semantic sense
- Word Vectors:
  - One-hot vectors:
    - Simple without implied ordering (an advantage over integers)
    - But the vectors are very large and sparse
    - Without encoding any meaning
  - Word Embedding Vectors (Word Embeddings):
    - Low dimensionality (less than vocabulary size $V$) AND
    - Embedded meaning, i.e
      - *semantic distance*, e.g. $\text{forest} \approx \text{tree}, \text{forest}$, and
      - *analogies*, e.g. $\text{Paris} : \text{France} :: \text{Rome} : \text{Italy}$

### How to Create Word Embeddings

To create word embeddings you always need a corpus of text, and an embedding method.

The context of a word tells you what type of words tend to occur near that specific word. The context is important as this is what will give meaning to each word embedding.

Word embeddings are a form **self-supervised learning**: Whilst the training data set is not labelled (as in unsupervised learning), it does contain data to supervise the learning process. This combination of supervised and unsupervised learning is self-supervised learning.

### Word Embedding Methods

Basic:

- word2vec (Google, 2013):
  - Continuous Bag-of-Words (CBOW)
  - Continuous Skip-Gram / Skip-Gram with Negative Sampling (SGNS)
- Global Vectors (GloVe; Stanford, 2014)
- fastText (Facebook, 2016):
  - Supports out-of-vocabulary (OOV) words

Advanced: Deep learning and contextual embeddings; tunable pre-trained models available:

- BERT (Google, 2018)
- ELMo (Allen Institute for AI, 2018)
- GPT-2 (OpenAI, 2018)

### Continuous Bag-of-Words Model

*Given context words as the input, predict the centre word as the output.*

### Cleaning and Tokenisation

- Letter case, lower / upper case
- Punctuation
- Numbers
- Special characters, e.g. $\nabla \rarr \emptyset$
- Special words, e.g. $:) \rarr \text{:happy:}$

```python
import nltk
from nltk.tokenize import word_tokenize
import emoji

nltk.download("punkt")

corpus = "..."

def tokenize(corpus):
    data = re.sub(r"[,!?;-]+", ".", corpus)
    data = nltk.word_tokenize(data)  # tokenize string to list of words
    data = [
        ch.lower() for ch in data
        if ch.isalpha()
        or ch == '.'
        or emoji.get_emoji_regexp().search(ch)
    ]
    return data

tokenize(corpus)
```

### Sliding Window of Words in Python

```python
def get_windows(words, C):
    """
    Args:
        words: sequence of words
        C: context size
    """
    i = C
    while i < len(words) - C:
        centre_word = words[i]
        context_words = words[(i-C) : i] + words[(i+1) : (i+C+1)]
        yield context_words, centre_word
        i += 1

for x, y in get_windows(["i", "am", "happy", "because", "i", "am", "learning"], 2):
    print(f"{x}\t{y}")

["I", "am", "because", "I"] "happy"
["am", "happy", "I", "am"]  "because"
["happy", "because", "am", "learning"]  "I"
```

### Transforming Words into Vectors

- Transform centre word into one-hot vector
- Transform context words into one-hot vectors
- Average context words vectors to give a single vector

```python
def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector

def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors
```

### CBOW Model Architecture

- $V$: size of the vocabulary
- $N$: size of the word embeddings
- $m$: size of the batch

---

- Input: Context words vector $x$ - dimension: $V \times m$
- Input layer - dimension $V \times m$
- ReLU activation applied to weights $W_1$ ($N \times V$), biases $b_1$ ($N \times m$)
- Hidden layer - dimension $N \times m$
- Softmax activation applied to weights $W_2$ ($V \times N$), biases $b_1$ ($V \times m$)
- Output layer: argmax - dimension $V \times m$
- Output: Centre word vector $\hat{y}$ - dimension $V \times m$

### CBOW Model Training

#### 1. Forward Propagation

$$Z_1 = W_1 X + b_1$$

$$H = \text{ReLU}(Z_1)$$

$$Z_2 = W_2 H + b_2$$

$$\hat{Y} = \text{softmax}(Z_2)$$

#### 2. Cost Function

Cross-Entropy Loss function:

$$J = - \sum_{k=1}^V{y_k \log{\hat{y}_k}}$$

Cost: mean of losses

$$
J_{\text{batch}} = - \frac{1}{m} \sum_{i=1}^m{J^{(i)}} = - \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{V} {y_j^{(i)} \log{\hat{y}_j^{(i)}}}
$$

#### 3. Backpropagation and Gradient Descent

Given $J_{\text{batch}} = f(W_1, W_2, b_1, b_2)$, calculate the partial derivatives of the cost with respect to the weights and biases, $\frac{\partial J_{\text{batch}}}{\partial W_1}, \frac{\partial J_{\text{batch}}}{\partial W_2}, \frac{\partial J_{\text{batch}}}{\partial b_1}, \frac{\partial J_{\text{batch}}}{\partial b_2}$, and update the weights and biases with the hyperparameter $\alpha$ in order to incrementally minimise the cost:

$$
W_i := W_i - \alpha \frac{\partial J_{\text{batch}}}{\partial W_i}
$$

$$
b_i := b_i - \alpha \frac{\partial J_{\text{batch}}}{\partial b_i}
$$

### Extracting Word Embedding Vectors

Three options: Extract the embedding at the index of a given word

- from weights matrix $W_1$
- from weights matrix $W_2$
- from both, and get their average

```python
W3 = (W1 + W2.T) / 2

for word in word2Ind:
    word_embedding_vector = W1[:, word2Ind[word]]

    word_embedding_vector = W2.T[:, word2Ind[word]]

    word_embedding_vector = W3[:, word2Ind[word]]

    print(f"{word}: {word_embedding_vector}")
```

### Evaluating Word Embeddings

Word embeddings can be evaluated intrinsically or extrinsically.

#### Intrinsic Evaluation

Test relationships between words by

- Analogy:
  - Semantic analogies, e.g. "France" is to "Paris" as "Italy" is to <?>
  - Syntactic analogies, e.g. "seen" is to "saw" as "been" is to <?>
- Clustering
- Visualisation

#### Extrinsic Evaluation

Test word embeddings on an external task, e.g. named entity recognition, parts-of-speech tagging

- Pro: Evaluates actual usefulness of embeddings
- Cons: Time-consuming and more difficult to trouble-shoot
