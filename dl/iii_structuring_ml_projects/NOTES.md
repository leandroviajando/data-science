# Structuring Machine Learning Projects

## Orthogonalisation

Orthoganility means being at 90 degrees to another.

Orthogonalisation is the process of isolating what to tune in order to try to achieve *one* effect; i.e. where knobs that can be tuned are orthogonal to each other, and every action will have only one effect.

1. Fit training set well on cost function
   - bigger network
   - Adam
2. Fit dev set well on cost function
   - regularisation
   - bigger training set
3. Fit test set well on cost function
   - bigger dev set
4. Performs well in real world
   - change dev set (mismatched distribution) or cost function

## Setting Up Your Goal

### Use a Single-Number Evaluation Metric

If you have more than one metric, how do you decide?

For example, if you want to decide based on precision and recall, you can use the F1 score which takes into account the precision-recall trade-off.

### Satisficing and Optimizing Metrics

For example, accuracy is an optimising metric - to be maximised; running time is a satisficing metric - it just needs to be good enough.

### Train / Validation / Test Set Distributions

It is not a problem to have different training and dev distributions.

For example, *you can use synthetic or augmented data as part of the training set.*

However, **validation and test sets must come from the same distribution.**

*Choose validation and test sets sampled from the same distribution, reflecting data you expect to get in the future and consider important to do well on.*

### Validation and Test Set Sizes

Set the validation and test set sizes big enough, and use as much data as possible for training.

## Comparing to Human-Level Performance

Knowing how well humans can do on a task is useful for deciding on whether to reduce bias or variance.

**Bayes' error**, also known as Bayes' optimal error or Bayes' risk, *is the lowest possible error rate that can be achieved by a classifier when the true class probabilities are known.* It represents the irreducible error rate that exists due to the inherent uncertainty and variability of the data.

A learning algorithm's performance can be better than human-level performance but it can never be better than Bayes' error.

### Avoidable Bias

Human-level performance can be used as a proxy for or measure of Bayes' Error.

Worse-than-human-level performance thus signifies "avoidable bias"; however, once human-level performance is surpassed, it may not be possible to improve performance much more.

### Improving Model Performance

First of all, spend a few days training a basic model and see what mistakes it makes.

## Error Analysis

Examine examples on which the algorithm made mistakes.

### Cleaning Up Incorrectly Labelled Data

### Build your First System Quickly, then Iterate

## Mismatched Training and Validation / Test Sets

### Addressing Data Mismatch

- Carry out manual error analysis to try to understand the differences between training, validation and test sets
- Collect more training data similar to validation and test sets

## Learning from Multiple Tasks

### Transfer Learning

Transfer learning makes sense when

- tasks A and B have the same input x
- you have a lot more data for task A than for task B

### Multi-Task Learning

Unlike softmax regression, one image can have multiple labels. This is called multi-label classification.

Note that the the output layer will therefore have a sigmoid activation, which will be applied to the multi-valued output labels; and that there may be entries that are unlabelled as those can simply be ignored when calculating the loss:

$$
\begin{align}
  y^{(i)} &= \begin{bmatrix}
      1 \\
      ? \\
      ? \\
      1 \\
      0
    \end{bmatrix}
\end{align}
$$

Such multi-task learning may result in better performance than learning multiple tasks separately: symmetrically, every one of the other 99 tasks can provide some data or provide some knowledge that helps every one of the other tasks in the list of 100 tasks.

It thus makes sense

- training on a set of tasks that could benefit from having shared lower-level features
- usually if the amount of data you have for each task is quite similar
- if can train a big enough neural network to do well on all the tasks

## End-to-End Deep Learning

Instead of breaking down a multi-stage problem into its components, just learn the task end-to-end.

Pros:

- Let the data speak
- Less hand-designing of components needed

Cons:

- May need large amounts of data
- Excludes potentially useful hand-designed components

**Key question: Do you have sufficient data to learn a function of the complexity needed to map `x` to `y`?**
