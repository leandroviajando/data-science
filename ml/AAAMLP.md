# [Approaching (Almost) Any Machine Learning Problem](https://github.com/amanpreetsingh459/AAAMLP)

## Supervised vs unsupervised learning

If the target is categorical, the problem becomes a classification problem. And if the target is a real number, the problem is defined as a regression problem. When we do not have any information, the problem becomes an unsupervised problem. It's also possible to convert a supervised dataset to unsupervised to see what it looks like when plotted.

## Cross-validation

If it's a standard classification problem, choose stratified k-fold.

For example, you might have multiple images of the same person in the training dataset. So, to build a good cross-validation system here, you must have stratified k-folds, but you must also make sure that patients in training data do not appear in validation data. Fortunately, scikit-learn offers **`GroupKFold`** for this purpose. It needs to be combined manually with `StratifiedKFold`.

If we have a large amount of data, we can opt for a hold-out based validation. Hold-out is also used very frequently with time-series data.

For regression problems, simple k-fold cross-validation works. However, if you see that the distribution of targets is not consistent, you can use stratified k-fold.

Sturge's Rule: `Number of bins` = $1 + \log_2(N)$

## Evaluation metrics

### Classification

- Accuracy = $(TP + TN) / (TP + TN + FP + FN)$
- Precision (P) = $TP / (TP + FP)$
- Recall (R) / sensitivity / True Positive Rate (TPR) = $TP / (TP + FN)$
- F1 score (F1) = $2PR / (P + R) = 2TP / (2TP + FP + FN)$ is a weighted average (harmonic mean) of precision and recall
- False Positive Rate (FPR) = $FP / (TN + FP)$
- Specificity / True Negative Rate (TNR) = $1 - FPR$
- Area under the ROC (Receiver Operating Characteristic) curve or simply AUC (AUC) is a plot with the TPR on the y-axis and the FPR on the x-axis, which allows finding a threshold based on whether you want a higher FPR or TPR (usually the top-left value on ROC curve should give quite a good threshold)
- Log loss = $-1.0 \times (target \times \log(prediction) + (1-target) \times \log(1-prediction))$ penalises incorrect / far-off predictions a lot, i.e. punishes you for being very sure and very wrong

With skewed data, using ROC / AUC allows as a metric allows for combining precision and recall in one metric - which in the case of skewed / imbalanced data is handier.

#### Multi-class classification

- Macro averaged precision / recall / ... calculates metrics for all classes individually and then averages them
- Micro averaged precision / recall / ... calculates class-wise true positive and false positive values and then calculates the overall metric
- Weighted precision / recall / ... is a macro metric using a weighted average on the number of items in each class

### Multi-label classification

- Precision at k (P@k): TODO
- Average precision at k (AP@k): TODO
- Mean average precision at k (MAP@k): TODO

### Regression

- Mean absolute error (MAE) = $Mean(Abs(True Value - Predicted Value))$
- Mean squared error (MSE) = $(True Value - Predicted Value)^2$
- Root mean squared error (RMSE) = $Sqrt(MSE)$
- Root mean squared logarithmic error (RMSLE)
- Mean percentage error (MPE) = $Mean(((True Value - Predicted Value) / True Value) * 100)$
- Mean absolute percentage error (MAPE)
- Coefficient of determination $R^2 = 1 - \frac{\sum_{i=1}^N{(y_{t_i} - y_{p_i})^2}}{\sum_{i=1}^N{(y_{t_i} - y_{t_{mean}})}}$

## Arranging machine learning projects

```txt
- data
- src
- notebooks
- models
```

## Approaching categorical variables

### Type os categorical variables

- Nominal variables don't have any order associated with them
- Ordinal variables
- Binary variables
- Cyclic variables, e.g. days in a week

### Encoding

- Tree-based models -> no need to normalise -> label encoding
- Linear, maximum margin or neural network models -> vital to normalise (/ standardise) -> one hot encode (or binarise)

scikit-learn's `LabelEncoder` can be used to encode categorical variables, and can be used directly in many tree-based models (decision trees, random forests, XGBoost, GBM, LightGBM).

Linear models, support vector machines or neural networks as they expect data to be normalised, or standardised. For these types of models, we can binarise or one-hot encode (`OneHotEncoder`) the data, and store it in a sparse format for memory efficiency.

Finally, with **entity embeddings**, you have an embedding layer for each categorical feature. So, every category in a column can now be mapped to an embedding (like mapping words to embeddings in natural language processing). You then reshape these embeddings to their dimension to make them flat and then concatenate all the flattened inputs embeddings. Then add a bunch of dense layers and an output layer.

## Feature engineering

A simple way to generate many features is just to create a bunch of polynomial features.

**Binning** enables you to treat numerical features as categorical, and you can use both the bin and the original feature.

*If you ever encounter missing values in categorical features, treat it as a new category!*

With numerical data, you can impute missing data, e.g. with the mean, median, `KNNImputer`, etc. *Imputing values for tree-based models is unnecessary as they can handle it themselves!*

When using linear models, always remember to scale or normalise features!

## Feature selection

Having too many features poses a problem well known as the curse of dimensionality. If you have a lot of features you must also have a lot of training samples to capture all the features.

The simplest form of selecting features would be to *remove features with very low variance.* If the features have a very low variance, they are close to being constant and thus, do not add any value to any model at all. It would just be nice to get rid of them and hence lower the complexity.

We can also *remove features with a high Pearson correlation* (`df.corr()`).

### Univariate feature selection

Univariate feature selection is a scoring of each feature against a given target.

Mutual information, ANOVA F-test, and Chi-Squared are some of the most popular methods for univariate feature selection.

There are two ways of using these in scikit-learn:

- `SelectKBest` keeps the top-k scoring features
- `SelectPercentile` keeps the top features which are in a percentage specified by the user

Chi-Squared is a particularly useful feature selection technique in NLP when we have a bag of words or tf-idf based features. It must be noted that Chi-Squared can only be used with non-negative data.

### Model-based feature selection

Univariate feature selection may not always perform well. Most of the time, people prefer doing feature selection using a machine learning model.

Greedy feature selection iteratively evaluates the loss/score of using each feature. The computational cost of this kind of method is very high.

Recursive Feature Elimination (RFE; available with scikit-learn) starts with all features and keeps removing one feature in every iteration that provides the least value to a given model. This works with models like SVM or logistic regression, where we get a coefficient for each feature which deicdes the importance of the features, and tree-based models, where we get feature importance in the form of the model coefficients.

Using L1 (Lasso) regularisation, most coefficients will be zero or close to zero, and we select the features with non-zero coefficients.

You can choose features with one model, and then train another model with those features. For example, you can use logistic regression coefficients to select the features, and then train a random forest with those features. Scikit-learn offers a `SelectFromModel` class that helps choose features directly from a given model.

## Hyperparameter optimisation

Hyperparameters control the training/fitting process of the model.

- Grid search
- Random search

When you create large models or introduce a lot of features, you also make it susceptible to overfitting the training data. To avoid overfitting, you need to introduce noise in training data features or penalise the cost function. This penalisation is called regularisation and helps with generalising the model.

In linear models, the most common types of regularisations are L1 (Lasso) and L2 (Ridge). With neural networks, we use dropout, data augmentation, noise, etc.

| Model | Hyperparameters | Range of values |
| ----- | --------------- | --------------- |
| Linear Regression | `fit_intercept` | `True`/`False` |
| Linear Regression | `normalize` | `True`/`False` |
| Ridge Regression | `alpha` | $0.01$, $0.1$, $1.0$, $10$, $100$ |
| Ridge Regression | `fit_intercept` | `True`/`False` |
| Ridge Regression | `normalize` | `True`/`False` |
| kNN | `n_neighbors` | $2$, $4$, $8$, $16$ |
| kNN | `p` | $2$, $3$ |
| SVM | `C` | $0.001$, $0.01$ .. $10$ .. $100$ .. $1000$ |
| SVM | `gamma` | `"auto"`, `"RS*"` |
| SVM | `class_weight` | `"balanced"`, `None` |
| Logistic Regression | `penalty` | L1, L2 |
| Logistic Regression | `C` | $0.001$, $0.01$ .... $10$ .. $100$ |
| Lasso Regression | `alpha` | $0.1$, $1.0$, $10$ |
| Lasso Regression | `normalize` | `True`/`False` |
| Random Forest | `n_estimators` | $120$, $300$, $500$, $800$, $1200$ |
| Random Forest | `max_depth` | $5$, $8$, $15$, $25$, $30$, `None` |
| Random Forest | `min_samples_split` | $1$, $2$, $5$, $10$, $15$, $100$ |
| Random Forest | `min_samples_leaf` | $1$, $2$, $5$, $10$ |
| Random Forest | `max_features` | `"log2"`, `"sqrt"`, `None` |
| XGBoost | `eta` | $0.01$, $0.015$, $0.025$, $0.05$, $0.1$ |
| XGBoost | `gamma` | $0.05-0.1$, $0.3$, $0.5$, $0.7$, $0.9$, $1.0$ |
| XGBoost | `max_depth` | $3$, $5$, $7$, $9$, $12$, $15$, $17$, $25$ |
| XGBoost | `min_child_weight` | $1$, $3$, $5$, $7$ |
| XGBoost | `subsample` | $0.6$, $0.7$, $0.8$, $0.9$, $1.0$ |
| XGBoost | `colsample_bytree` | $0.6$, $0.7$, $0.8$, $0.9$, $1.0$ |
| XGBoost | `lambda` | $0.01-0.1$, $1.0$, `"RS*"` |
| XGBoost | `alpha` | $0$, $0.1$, $0.5$, $1.0$, `"RS*"` |

## Approaching image classification & segmentation

An image is a matrix of number, a two-dimensional matrix with values ranging from $0$ to $255$, where $0$ is black and $255$ is black. (If you are dealing with RGB images, then you have three matrices instead of one. But the idea is the same.)

Let's consider a classic case of skewed binary image classification. Therefore, we choose the evaluation metric to be AUC and go for a stratified k-fold cross-validation scheme.

**Filters** are two-dimensional matrices which are initialised by a given function. He Initialisation is a good choice for CNNs because most modern networks use ReLU activation functions and proper initialisation is required to avoid vanishing gradients.

**Convolutions** are a summation of elementwise multiplication (cross-correlation) between the filter and the pixels it is currently overlapping in a given image.

**Padding** is a way of keeping image size the same.

With **dilation**, you skip some pixels in each convolution. This is particularly effective in segmentation tasks.

In a segmentation task we try to remove/extract foreground from background. The most ppopular model used for segmentation is U-Net, which has two parts: an encoder and a decoder.

**Pooling** is a way of down-sampling the image. *Max pooling detects edges, and average pooling smoothens the image.*

ResNet consists of **residual blocks** that transfer the knowledge from one layer to further layers by skipping some layers in between. These kinds of connections of layers are known as **skip-connections** since we are skipping one or more layers. Skip-connections help with the vanishing gradient issue by propagating the gradients to further layers, which allows us to train very large CNNs without loss of performance.

As a general rule of thumb, choose the Adam optimiser, use a low learning rate, reduce the learning rate on a plateau of validation loss, try some data augmentation, try preprocessing the images (e.g. cropping if needed; this can also be considered preprocessing), change the batch size, etc.

## Approaching text classification/regression

Splitting a string into a list of words is called **tokenisation**.

One of the basic models you should always try with a classification problem in NLP is **bag of words**, a huge sparse matrix that stores counts of all the words in the corpus.

scikit-learn's `CountVectorizer` first tokenises the sentence and then assigns a value to each token. So, each token is represented by a unique index. These unique indices are the columns that we see.

**Term frequencies - inverse document frequency (TF-IDF)** are available in scikit-learn as `TfidfVectorizer` and `TfidfTransformer`:

$$
TF(t) = \frac{\text{Number of times a term t appears in a document}}{\text{Total number of terms in the document}}
$$

$$
IDF(t) = \log(\frac{\text{Total number of documents}}{\text{Number of documents with term t in it}})
$$

$$
TF-IDF(t) = TF(t) \times IDF(t)
$$

**N-grams** are combinations of words in order, and there is an implementation available in the `nltk` package.

When we calculate counts or tf-idf, we consider one n-gram as one entirely new token. So, in a way, we are incorporating context to some extent. Both `CountVectorizer` and `TfidfVectorizer` implementations of scikit-learn have a parameter `ngram_range`, which has a minimum and maximum limit.

**Stemming and lemmatization** reduce a word to its smallest form.

Lemmatization is more aggressive than stemming. When we do stemming, we are given the smallest form of a word, which may or may not be a word in the dictionary for the language the word belongs to. However, in the case of lemmatization, this will be a word.

**Topic extraction** can be done using non-negative matrix factorization (NMF) or latent semantic analysis (LSA), which is also popularly known as singular value decomposition (SVD). These are decomposition techniques that reduce the data to a given number of components. You can fit any of these on a sparse matrix obtained from `CountVectorizer` or `TfIdfVectorizer`.

To be able to use deep learning, we need to use **word embeddings**. Until now, we converted tokens into numbers. So, if there are $N$ unique tokens in a given corpus, they can be represented by integers ranging from $0$ to $N-1$. Now we will represent these integer tokens with vectors. This representation of words into vectors is known as word embeddings or word vectors. Google's Word2Vec is one of the oldest approaches to convert words into vectors. We also have FastText from Facebook and GloVe (Global Vectors for Word Representation) from Stanford. These approaches are quite different from one another.

The basic idea is to build a shallow network that learns the mbeddings for words by reconstruction of an input sentence. So, you can train a network to predict a missing word by using all the words around and during this process, the network will learn and update embeddings for all the words involved. This approach is also known as **Continuous Bag of Words (CBoW)** model. You can also try to take one word and predict the context words isntead. This is called **skip-gram** model. Word2Vec can learn embeddings with both methods.

Text data is very similar to time-series data. Any sample in our reviews is a sequence of tokens at different timestamps which are in increasing order, and each token can be represented as a vector/embedding.

This means we can use models that are widely used for time series data, such as Long-Short-Term Memory (LSTM) or Gated Recurrent Units (GRU) or even Convolutional Neural Networks (CNNs).

The more words for which you have pre-trained embeddings, the better are the results! A trick is to learn the embedding layer, i.e. make it trainable and then train the network.

**Transformer**-based networks (BERT, RoBERTa, XLNet, XLM-RoBERTa, T5) are able to handle dependencies which are long-term in nature. LSTM looks at the next word only when it has seen the previous word. This is not the case with transformers. They can look at all the words in the whole sentence simultaneously. Due to this, one more advantage is that it can easily be parallelised and uses GPUs more efficiently. Still, these models are very hungry in terms of computational power needed to train them. So, if you do not have a high-end system, it might take much longer to train a model compared to LSTM or TF-IDF based models.

## Approaching ensembling and stacking

## Approaching reproducible code & model serving
