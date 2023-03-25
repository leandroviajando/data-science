# [Natural Language Processing with Classification and Vector Spaces](https://www.coursera.org/learn/classification-vector-spaces-in-nlp?specialization=natural-language-processing)

## 1.1. Sentiment Analysis with Logistic Regression

### Vocabulary

A **vocabulary** is a set of unique words (from a list of sentences, tweets, etc.).

### Feature Extraction

**Feature extraction** can be used to convert the text into features, e.g. a vector of zeros and ones.

This type of representation with a small relative number of non-zero values is called a **sparse representation**.

Such sparse representations can be problematic because it is much harder for a model to learn enough so that it could generalize well on the test set.

### Frequency Counts

Divide the text corpus into two classes, i.e. positive and negative, and count each time each word appears in either class. This gives a dictionary mapping from each instance to its (positive and / or negative) frequency.

**Frequencies** *are a form of feature extraction, and can be represented as a vector*. For example, given the text `I am happy because I am learning NLP @deeplearning`,  we would end up with a **feature vector** `[1, 4, 2]` where the first index is the `bias`, the second index the number of `positive words`, and the third index the number of `negative words` in the given instance.

*Relative to a feature vector of size* $n$*, a feature vector of size* $3$ *will result in much more efficient training and prediction times.*

### Preprocessing

- Remove stopwords, punctuation, handles and URLs
- Stemming: reducing a word to its stem
- Lower-casing

### Logistic Regression

Logistic regression makes use of the sigmoid function which outputs a probability between $0$ and $1$. The sigmoid function with some weight parameter $\theta$ and some input $x(i)$ is defined as:

$$h(x^{(i)}, \theta) = \frac{1}{1 + e^{-\theta^T x^{(i)}}}$$

Note that as $\theta^T x^{(i)}$ gets closer and closer to $-\infty$, the denominator of the sigmoid function gets larger and larger and, as a result, the sigmoid gets closer to $0$.

On the other hand, as $\theta^T x^{(i)}$ gets closer and closer to $\infty$ the denominator of the sigmoid function gets closer to $1$ and as a result the sigmoid also gets closer to $1$.

Now given a tweet, you can transform it into a vector and run it through your sigmoid function to get a prediction.

#### Training

1. Initialize parameters $\theta$
2. Classify $h = h(X, \theta)$
3. Get gradient $\nabla = \frac{1}{m} X^T (h-y)$
4. Update $\theta = \theta - \alpha \nabla$
5. Get loss $J(\theta)$
6. Repeat from step 2 *until the cost converges*

#### Testing

Test the model's accuracy *on unseen data* - the validation set.

#### Cost Function

Logistic regression uses the **binary cross-entropy** cost function.

The formula can be split into two components:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m}{[ y^{(i)} log(h(x^{(i)}, \theta)) + (1-y^{(i)}) log(1-h(x^{(i)}, \theta)) ]}$$

which is derived from the probability:

$$P(y | x^{(i)}) = h(x^{(i)}, \theta)^{y^{(i)}} (1 - h(x^{(i)}, \theta))^{(1-y^{(i)})} $$

- Strong disagreement = high cost ($+/-\infty$)
- Strong agreement = low cost

If $y=1$ and you predict something close to $0$, you get a cost close to $\infty$.

The same applies for $y=0$ and a prediction close to $1$.

On the other hand, if you get a prediction equal to the label, you get a cost of $0$.

In either case, the objective is to minimize $J(θ)$.

Question: For what value of $\theta^T x$ in the sigmoid function does $h(x^{(i)}, \theta) = 0.5$? $0$

## 1.2. Sentiment Analysis with Naive Bayes

Simple, fast and robust!

### Bayes' Rule

![Conditional Probabilities](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xoYzIfcHT0uGMyH3B69LQw_3b10c11c4a504416b8ea133f0c974995_Screen-Shot-2020-09-08-at-1.23.56-PM.png?expiry=1674432000000&hmac=5h2fvwn76byXKq1exQvOFSmeeA24TF5cVH6K4kE_7wM)

$$P(X | Y) = \frac{P(X \cap Y)}{P(Y)}$$

$$P(Y | X) = \frac{P(Y \cap X)}{P(X)}$$

$$P(Y \cap X) = \frac{P(X | Y)}{P(Y)} = \frac{P(Y | X)}{P(X)}$$

$$P(X | Y) = P(Y | X) \times \frac{P(X)}{P(Y)}, P(Y | X) = P(X | Y) \times \frac{P(Y)}{P(X)}$$

### Naive-Bayes Classification

Naive Bayes inference condition rule for binary classification:

$$\prod_{i=1}^{m} \frac{P(w_i | pos)}{P(w_i | neg)} > 1$$

#### Laplacian Smoothing

If a word does not appear in the training data, it automatically gets a probability of zero.

To avoid $P(w_i | class) = 0$,

$$P(w_i | class) = \frac{freq(w_i, class)}{N_{class}} => P(w_i | class) = \frac{freq(w_i, class) + 1}{N_{class} + V_{class}}$$

where $class \in \{pos, neg\}$, $N_{class}$ is the frequency of all words in the class, and $V_{class}$ is the number of unique words in the class.

#### Log Likelihood

$$
\text{Word Sentiment} =
\begin{cases}
    ratio(w) = \frac{P(w | pos)}{P(w | neg)} \\
    \lambda(w) = \log{\frac{P(w | pos)}{P(w | neg)}}
\end{cases}
$$

The higher the ratio, the more positive a word is.

To do inference, calculate

$$\frac{P(w | pos)}{P(w | neg)} \prod_{i=1}^{m} \frac{P(w_i | pos)}{P(w_i | neg)} > 1$$

As `m` grows larger, this can cause numerical overflow issues. To avoid any such problems, take the log of the formula:

$$\log( \frac{P(w | pos)}{P(w | neg)} \prod_{i=1}^{m} \frac{P(w_i | pos)}{P(w_i | neg)} ) = \log{\frac{P(w | pos)}{P(w | neg)}} + \sum_{i=1}^{m}{\log{\frac{P(w_i | pos)}{P(w_i | neg)}}} > 0$$

The first component is the log prior and the second component is the log likelihood.

This can be used for inference; if the expression is positive, the tweet is predicted to be positive, or else negative.

(The log of two probabilities is bounded by negative and positive infinity.)

*Note that the log prior will be zero for balanced datasets.*

#### Training

0. Collect and annotate corpus
1. Preprocess:
   - Lowercase
   - Remove punctuation, handles, names
   - Remove stopwords
   - Stemming
   - Tokenize sentences
2. $freq(w, class)$
3. $P(w | pos), P(w | neg)$
4. $\lambda(w)$
5. Log prior - will be zero for balanced datasets, but **important for imbalanced datasets!**

#### Testing

Performance on unseen data.

Need $X_{val}, Y_{val}, \lambda, \text{log-prior}$

#### Applications

- Sentiment analysis

$$P(pos | tweet) \approx P(pos) P(tweet | pos)$$

$$P(neg | tweet) \approx P(neg) P(tweet | neg)$$

$$\frac{P(pos | tweet)}{P(neg | tweet)} = \frac{P(pos)}{P(neg)} \prod_{i=1}^{m}{\frac{P(w_i | pos)}{P(w_i | neg)}}$$

- Spam filtering

$$\frac{P(spam | email)}{P(non-spam | email)}$$

- Information retrieval

$$P(document_k | query) \infty \prod_{i=0}^{|query|}{P(query_i | document_k)}$$

Retrieve document if $P(document_k | query) > threshold$.

- Word disambiguation

$$\frac{P(river | text)}{P(money | text)}$$

#### Assumptions

**Independence of words in a sentence!** *Naive Bayes is called Naive because it naively assumes independence.* For example, the words sunny and hot tend to depend on each other and are correlated to a certain extent with the word "desert". Naive Bayes assumes independence throughout, however.

**Relative frequencies:** For example, there may be more positive than negative tweets (e.g. due to content moderation) and this will affect the model!

#### Error Analysis

- **Processing as a Source of Errors**: Removing punctuation and stop words may cause parts of the context being lost, e.g. `:(`. Word order matters!

- **Adversarial attacks**: Sarcasm, irony and euphemisms

## 1.3. Vector Space Models

Vector spaces are a representation of words and documents as vectors that captures **relative meaning** useful for information extraction, machine translation, chatbots, etc.

### Similarity measures

Counts of occurence can be used to measure similarity between words / documents:

- **Word by word (W/W):** Number of times two words occur together within a certain distance `k`.
- **Word by document (W/D):** Number of times a word occurs within a certain category.

**Euclidean distance** is the norm of the difference between vectors.

$$||\vec{v}|| = \sqrt{\sum_{i=1}^{n}{v_i^2}}$$

$$\vec{v} \cdot \vec{w} = \sum_{i=1}^{n}{\vec{v_i} \cdot \vec{w_i}}$$

```python
v = np.array([1, 6, 8])
w = np.array([0, 4, 6])

d = np.linalg.norm(v - w)
```

Euclidean distance is not a good measure if the documents are of different sizes: Two documents of similar size may appear more similar to each other than to another document of different size, just due to the difference in size (which affects the length of the vectors).

**Cosine similarity** *is a better measure when corpora are of different sizes* because the angle is not dependent on the size of the copora. It gives values between $0$ ($\beta = 0^{\circ}$, i.e. similar) and $1$ ($\beta = 90^{\circ}$, i.e. dissimilar).

$$cos(\beta) = \frac{\vec{v} \cdot \vec{w}}{||\vec{v}|| ||\vec{w}||}$$

```python
a = np.array([1, 0, -1, 6, 8])
b = np.array([0, 11, 4, 7, 6])

cosine_similarity = np.dot(a, b) / ( np.linalg.norm(a) * np.linalg.norm(b) )
```

### Dimensionality Reduction

**PCA** is an algorithm used for dimensionality reduction that can find correlated features in your data. It's very helpful for visualizing your data to check if your representation captures relationships among words. Given any $d$-dimensional vector, you can transform it into two dimensions and create a plot.

- **Eigenvectors** give the direction of uncorrelated features.
- **Eigenvalues** are the variance of the new features/ the amount of information retained by each feature.
- The dot product gives the projection on uncorrelated features.

**PCA steps:**

1. Mean-normalize data.
2. Compute the covariance matrix.
3. Compute `SVD` on your covariance matrix. This returns $[U S V] = \text{svd}(\Sigma)$. The matrix `U` is labelled with eigenvectors, and `S` is labelled with eigenvalues.
4. Multiply with the first `n` columns of vector `U`, i.e. $X U[:, 0:n]$.

#### [The Rotation Matrix](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/supplement/fwEUM/the-rotation-matrix-optional-reading)

If you want to rotate a vector $r$ with coordinates $(x, y)$ and angle $\alpha$ counterclockwise over an angle $\beta$ to get vector $r'$ with coordinates $(x', y')$ then the following holds:

$$x = r \times \cos(\alpha)$$

$$y = r \times \sin(\alpha)$$

$$x' = r' \times \cos(\alpha + \beta)$$

$$y' = r' \times \sin(\alpha + \beta)$$

Trigonometric addition gives:

$$\cos(\alpha + \beta) = \cos(\alpha) \cos(\beta) - \sin(\alpha) \sin(\beta)$$

$$\sin(\alpha + \beta) = \cos(\alpha) \sin(\beta) + \sin(\alpha) \cos(\beta)$$

As the length of the vector stays the same:

$$x' = r \times \cos(\alpha) \cos(\beta) - r \times \sin(\alpha) \sin(\beta)$$

$$y' = r \times \cos(\alpha) \sin(\beta) + r \times \sin(\alpha) \cos(\beta)$$

Which equates to:

$$x' = x \times \cos(\beta) - y \times \sin(\beta)$$

$$y' = x \times \sin(\beta) + y \times \cos(\beta)$$

Thus,

$$
\begin{bmatrix}
x' \\
y' \\
\end{bmatrix}
=
\begin{bmatrix}
\cos(\beta) & - \sin(\beta) \\
\sin(\beta) & \cos(\beta) \\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
\end{bmatrix}
$$

Similarly, clockwise rotation can be expressed as

$$
\begin{bmatrix}
x' \\
y' \\
\end{bmatrix}
=
\begin{bmatrix}
\cos(\beta) & \sin(\beta) \\
- \sin(\beta) & \cos(\beta) \\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
\end{bmatrix}
$$

## 1.4. Machine Translation and Document Search

### Transforming word vectors

Translation = transformation

Transformation matrix $\bold{R}$ to map vector from one vector space (language, $\bold{X}$) to another ($\bold{Y}$)

$$
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
2 & 0 \\
0 & -2 \\
\end{bmatrix}
=
\begin{bmatrix}
2 & -2
\end{bmatrix}$$

```python
np.dot(X, R)
```

$$\bold{X} \bold{R} \approx \bold{Y}$$

In order to get a transformation of `X` as similar to `Y` as possible, *minimize the distance between `XR` and `Y`:*

The Loss is defined as the **Frobenius norm:** $|| \bold{X} \bold{R} - \bold{Y} ||_F$

The norm is a measure of the size or magnitude of a vector. It quantifies the length of a vector by considering the squares (and possibly square roots) of its individual elements and summing them up. Different types of norms, such as the Euclidean norm and the Manhattan norm, capture different notions of vector magnitude.

The Frobenius norm is a specific norm used for matrices. It quantifies the size or magnitude of a matrix by considering the squares of all its elements and summing them up. It is analogous to the Euclidean norm for vectors. The Frobenius norm provides a measure of how "spread out" or "large" the matrix is.

The Frobenius norm of a matrix $A$ is calculated as:

$$||A||*F = \sqrt{\Sigma*{i=1}^{m} \Sigma_{j=1}^{n} |a_{ij}|^2}$$

For example, $||\begin{bmatrix}2 & 2 \\ 2 & 2\end{bmatrix}||_F = \sqrt{2^2 + 2^2 + 2^2 + 2^2} = 4$

```python
A = np.array([[2, 2], [2, 2]])

F = np.sqrt(np.sum(np.square(A)))
```

*In practice, it's easier to work with the square of the Frobenius norm because taking its derivative is easier,* i.e. the objective is to minimise $|| \bold{X} \bold{R} - \bold{Y} ||_F^2$

#### Gradient descent

1. Initialize `R`
2. Create a for loop, and inside the loop:
3. Compute the gradient `g`
4. Update the loss

$\bold{R} = \bold{R} - \alpha g$ where the gradient $g = \frac{d}{dR} Loss = \frac{2}{m} \bold{X}^T (\bold{X} \bold{R} - \bold{Y})$

### kNN

After you have computed the output of $XR$ you get a vector. You then need to find the most similar vectors to your output.

This is where the **kNN** algorithm comes in handy.

### Hash tables - Dividing vector spaces into regions

Hash tables are data structures based on dictionaries that allows to index data in order to improve heavy lookup tasks.

Hash tables can be used to improve performance of the kNN algorithm by putting similar instances in the same bucket in the hash table (e.g. divide countries by continent, cities by country or similar).

Now, given an input, you don't have to compare it to all the other examples, you can just compare it to all the values in the same hash bucket that input has been hashed to.

Hash tables are useful

- to speed up the time it takes when comparing similar vectors
- to not have to spend time comparing vectors with other vectors that are completely different

Hash tables are useful because they

- allow us to divide vector spaces into regions
- speed up look up
- can always be reproduced

### Locality Sensitive Hashing

A hashing function sensitive to the locality of a vector.

A dot product is a projection of another vector (and its magnitude) onto the given vector. It will be negative if the other vector points in the other direction, i.e. is on the other side.

Thus, the sign of the dot product gives hash values to separate instances by which side of the plane they fall on.

```python
def side_of_plane(P, v):
    dot_product = np.dot(P, v.T)

    sign_of_dot_product = np.sign(dot_product)

    return np.asscalar(sign_of_dot_product)
```

This can be extended to **multiple planes:**

$$hash = \Sigma_i^H{2^i \times h_i}$$

where $h_i = 1$ if $sign_i \ge 0$ or $0$ else

```python
def hash_multiple_plane(P_1, v):
    hash_value = 0

    for i, P in enumerate(P_1):
        sign = side_of_plane(P, v)

        hash_i = 1 if sign >= 0 else 0

        hash_value += pow(2, i) * hash_i

    return hash_value
```

### Approximate Nearest Neighbours

Taking advantage of multi-plane locality hashing thus makes the kNN algorithm much faster than naive / brute search.

### Searching documents

Documents, like any other text, can be embedded into vector spaces so that nearest neighbours refer to text with similar meaning:

1. Initialize the document embedding as an area of zeros.
2. Now, for each word in the document, you'll get the word vector if the word exists in the dictionary else zero.
3. Add this all up and return the documents embedding.

The look up for similar documents can be sped up with

- Approximate Nearest Neighbours
- Locality Sensitive Hashing
