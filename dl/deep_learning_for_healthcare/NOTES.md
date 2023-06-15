# Deep Learning for Healthcare

## 2.1 Embeddings

### One-Hot Encodings

One-hot encoding is commonly used to encode categorical data.

A one-hot encoding vector is a multi-dimensional vector of all 0s except for one dimension of 1.

Problem: The distance between any two one-hot encoding vectors is always the same. Therefore, no distinction can be made between similar and dissimilar words.

### Word2Vec

There are two different Word2Vec algorithms: skip-gram and CBOW.

A larger context window means that words that are farther apart will be considered similar in the skip gram calculation.

**Negative sampling** can be used to speed up Word2Vec: it treats the pairs of context words and the target word as positive examples, and pairs of random words and the input target word as negative examples.

Applications:

- **Similarity search:** Euclidean distance can be applied to embeddings to find similar concepts.
- **Algebra operations:** Embeddings can be subtracted or added to better understand concepts.
- **Features for predictive modelling:** Word2Vec embeddings can be used as input feature vectors to support downstream classification or regression models.

### t-SNE

PCA is a dimensionality reduction algorithm that tries to *preserve global distance* between all pairs of data points.

t-SNE tries to project high-dimensional data into 2D space while *preserving local distance*.

t-SNE is thus better suited for visualising high-dimensional data.

The t-SNE algorithm's input distribution utilises a Gaussian kernel. Its output distribution is the Student's t distribution with 1 degree of freedom. Its objective function is the KL divergence between input and output distributions.

## 2.2 Convolutional Neural Networks

Compared to fully-connected neural networks, CNNs have more **local connections**.

CNNs enable **weight sharing** *through filters*.

CNNs utilise **pooling** to try to address distortion / the translation invariance challenge.

They can be applied to images and text data, time series, ... any data that can be represented as a grid-like structure.

Its advantages are sparse interactions, parameter sharing, and translational invariance.

### Architectures

AlexNet was the first deep neural network that won the ImageNet classification challenge.

VGG introduced the idea of stacking smaller filters instead of a larger filter.

InceptionNet introduced parallel paths.

ResNet introduced skip connections to allow learning networks of variable depths.

## 2.3 Recurrent Neural Networks

## 2.4 Autoencoders

Autoencoders are a type of feedforward neural network that can be classified as an unsupervised, and a dimensionality reduction method.

- Unsupervised: no labels are required
- Data specific: compress similar data to the training data
- Lossy: reconstruction will not be identical to the input

Compression & decompression: It learns the latent representation of a given sample `x`; *minimising reconstruction error:*

$$
\begin{cases}
L(x, r) = \lVert x-r \rVert^2 & \text{for Gaussian input} \\
L(x, r) = - \sum_i [x_i \log{r_i} + (1-x_i) \log(1-r_i)] & \text{for binary input} \\
\end{cases}
$$

Variants:

- Sparse autoencoder
- Denoising autoencoder
- Stacked autoencoder

### Sparse autoencoder

### Denoising autoencoder

The **denoising autoencoder** adds random noise to the original input before applying the autoencoder model.

It is more robust to noise thanks to the introduction of corrupted inputs.

### Stacked autoencoder

A **stacked autoencoder** is a deep neural network of `2K` layers where `K` is the number of autoencoders.

It is trained sequentially as `K` separate autoencoders; applying multiple encoders first, and then applying the corresponding decoders in reverse order.

## 3.1 Attention Mechanism

An encoder-decoder sequence-to-sequence model's static context vector has limited capacity in modelling sequences.

The attention mechanism makes use of a *dynamic context vector*.

## [3.2 Graph Neural Networks](https://antoniolonga.github.io/Pytorch_geometric_tutorials/)

- [Tutorials](https://github.com/AntonioLonga/PytorchGeometricTutorial/tree/main)

Consider a graph $G = (V, E, X)$ with

- node set $V$
- edge set $E$
- adjacency matrix $A$
- node features $X$
  - embeddings
  - can also have no features, in that case just use one-hot encoding

Inputs:

- Nodes
- Edges
- Node features

Applications:

- Node classification, e.g. sick/healthy
- Link prediction
- Graph property prediction, i.e. if a molecule (represented as a graph) is safe or toxic
- Community detection

Challenges:

- Graphs can have arbitrary sizes
- No fixed node ordering
- Dynamic updates, i.e. adding or removing nodes
- Heterogeneous features, i.e. different types of features

The objective is to learn node embeddings on graphs.

**Node embeddings** map nodes to $d$-dimensional vectors such that similar nodes are close to each other. The embedding function is a deep neural network that leverages the graph structure.

To apply graph convolutions, one has to consider neighbouring nodes, rather than neighbouring pixels in the grid.

To achieve efficient training, various sampling strategies are introduced:

- Aggregation: The direct neighbours of a node make up its aggregation set.
- Layering: Embeddings can have multiple layers
  - Layer 0: the node itself
  - Layer 1: $h^{(1)}$ aggregate one-hop neighbours
  - Layer k: $h^{(k)}$ aggregate k-hop neighbours

To compute the embeddings at layer $l+1$, add the aggregate neighbours' embeddings:

$$
h_i^{(l+1)} = \sigma(h_i^{(l)} W^l + \sum_{j \in N_i}{h_j^{(l)} W^l})
$$

Steps:

1. Define a neighbourhood aggregation function
2. Define a loss function on the embeddings
3. Train on a set of nodes based on local neighbourhoods
4. Generate embeddings for any node based on node features

### Graph Convolutional Networks (GCN)

GCNs are an efficient method for generating node embeddings on graphs.

The number of parameters of the network is proportional to the graph size.

### Message-Passing Neural Networks (MPNNs)

The objective is to generate both node embeddings and graph embeddings.

MPNNs handle both node and edge features.

The **message-passing stage** computes the message $m_V$ for node $v$, and updates the node embedding $h_V$.

The **read-out operation**'s purpose is to generate an embedding $h_G$ for the entire graph.

Average or max pooling are common functions for the read-out operation.

**Limitation:** This type of network is computationally more expensive to train and can only handle smaller graphs such as molecule graphs.

### Graph Attention Networks (GAT)

The GCN architecture aggregates neighbourhoods uniformly with equal weight.

The GAT architecture enables assigning different levels of importance to different neighbours, using the attention mechanism.

Alignment weight $e_{ij}$ is computed between nodes $i$ and $j$:

$$
e_{ij} = \text{LeakyReLU}(a^T[W h_i \Vert W h_j ])
$$

The attention weight is only computed for neighbours (the other have $0$ attention):

$$
\alpha_{ij} = \text{Softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{v_j \in N_i}{}\exp(e_{ij})}
$$

**Multi-head attention:** This attention process can be repeated $R$ times. This is beneficial because it stabilises the learning process of the attention mechanism. The attention weight $\alpha_{ij}$ thus becomes a vector of size $R$.

In summary, GAT

- is a combination of the attention mechanism and the GCN architecture
- allows assigning different importance $\alpha_{ij}$ to different neighbours
- is computationally more efficient than MPNN but slower than GCN

## 3.3 Memory Networks & Self Attention

### Original Memory Networks

A memory network is a deep neural network with memory components:

- Input feature map $I$:

  - maps input $x$ to an internal feature representation $I(x)$
    - one-hot encoding
    - RNN if the inputs are sequences

- Generalisation $G$: updates old memories given new input $x$

$$
m_i = G(m_i, I(x), m) \forall i
$$

- Output feature map $O$:

  - computes output feature $o = O(I(x), m)$, e.g. the most relevant memory $m_i$

- Response $R$:

  - maps output to the final response $r = R(o)$, e.g.

    - softmax for classification
    - RNN for sequence generation

The generalisation module is the most innovative step in the original memory network model.

Limitation: not end-to-end trained because of argmax op for finding the
optimal memory slot

### End-to-End Memory Networks

Idea: replace the argmax in the original memory network with softmax with attention, so that the gradient can be computed at all steps.

Multi-layer end-to-end memory networks can be created via multiple embedding matrices for keys and values.

The model parameters are matrices $A, B, C \text{ and } W$.

### Self Attention

Three versions of embeddings of the input $X$ are produced: $Q, K, V$. The main reason for this is to be able to apply and learn those embeddings independently in parallel.

Multiple versions of self-attention are concatenated together to produce more robust embeddings, called multi-head self attention.

### Transformers

A Transformer is an effective embedding method for sequential data using the self-attention mechanism.

The ideas used in the Transformer encoder are:

- Self attention
- Positioning encoding
- Multi-head attention
- Residual connections

The ideas used in the Transformer decoder are:

- Masked attention
- Self attention
- Residual connections

### BERT

BERT provides a transformer-based, masked language model. It masks x% of input words and tries to predict them with other words.

## 3.4 Generative Models

### Generative Adversarial Networks

Goal: create new samples that resemble training data

Train two neural networks:

- The generator creates realistic but synthetic samples. Its input is random noise.
- The discriminator differentiates synthetic and real samples. Its inputs are synthetic and real examples.

Loss function:

$$
E_{x \sim p_{\text{data}(x)}}[\log{D(x)}] + E_{z \sim p_z}[z] (\log(1-D(G(z))))
$$

where

- the first term, sampled from real data, is the log-likelihood that real samples are real
- the second term, sampled from random noise, is the log-likelihood that fake samples are fake

The second term is the generator's loss, and both terms make up the discriminator's loss.

### Variational Autoencoders

VAEs are a generative technique for creating realistic data samples, combining deep learning and probabilistic graphical modelling.
