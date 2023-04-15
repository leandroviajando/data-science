# [Generative Adversarial Networks](https://www.coursera.org/specializations/generative-adversarial-networks-gans)

## 1.1 Introduction

**Discriminative models** distinguish between classes (classification): Use features $X$ to predict class $Y$ with probability $P(Y | X)$.

**Generative models** learn to produce examples: Given class $Y$ and noise $\xi$, predict features $X$ with probability $P(X | Y)$.

There are two types of generative models:

- Variational Autoencoders with an encoder-decoder architecture
- Generative Adversarial Networks consisting of a generator and a discriminator competing with each other

### Intuition

The Generator learns to make fakes that look real.

The Discriminator learns to distinguish real from fake.

They both learn from the competition with each other so at the end, fakes look real.

### Discriminator

The discriminator is a classifier. It learns the probability of class $Y$ (real or fake) given features X, $P(Y | X)$.

This conditional probability is the feedback for the generator.

### Generator

The generator produces fake data by learning the probabilities of features $X$.

It takes as input *noise*, i.e. random features.

### Cost Function

The discriminator minimises the cost. The generator maximises cost.

Binary Cross-Entropy:

- two parts (one relevant for each class)
- close to zero when the label and the prediction are similar
- approaches infinitiy when the label and the prediction are different

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m{ y^{(i)} \log{h(x^{(i)}, \theta)} + (1 - y^{(i)}) \log(1 - h(x^{(i)}, \theta)) }
$$

- $\theta$ = parameters
- $x$ = features
- $y$ = label
- $h$ = prediction
- $/m$ = average loss over the batch
- $-$ = ensures that the cost is always greater or equal to $0$

```python
import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, in):
        super().__init__()

        self.log_reg = torch.nn.Sequential(
            torch.nn.Linear(in, 1),
            torch.nn.Sigmoig(),
        )

    def forward(self, x):
        return self.log_reg(x)
```

### Training

GANs train in an alternating fashion. The generator and discriminator should always be at a similar "skill" level.

```python
model = LogisticRegression(16)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=.01)

for t in range(n_epochs):

    # Forward Propagation:
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Optimization:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 1.2 Deep Convolutional GANs (DCGANs)

### Activations

- *Differentiable* - for backpropagation
- *Non-linear* - to approximate complex features (else would just reduce to a single layer in linear regression)

### Common Activation Functions

- ReLU $= \max(0, z^{[[l]]}), [0, \infty)$
- Leaky ReLU $= \max(a z^{[[l]]}, z^{[[l]]})$
- Sigmoid $= \frac{1}{a + e^{-z^{[l]}}}, (0, 1)$
- Tanh $= \tanh(z^{[l]}), (-1, 1)$

ReLU activations suffer from **dying ReLU problem:** With a zero derivative some nodes get
stuck on the same value and their weights stops learning.

Leaky ReLU solves this.

Sigmoid and tanh both have vanishing gradient and saturation problems.

### Batch Normalisation

- smoothes the cost function
- reduces internal covariate shift
- speeds up learning

**Covariate shift** means that the distributions of some variables are dependent on another.

Normalise around mean at $0$ and standard deviation at $1$:

$$
\hat{z}*i^{[l]} = \frac{z_i^{[l]} - \mu*{z_i^{[l]}}}{\sqrt{\sigma^2_{z_i^{[l]}} + \epsilon}}
$$

The batch mean and standard deviation are used for training, while the running average statistics from all batches are used for testing.

Batch Normalisation additionally introduces learnable parameters to get the optimal distribution: $\gamma$ is the shift factor, and $\beta$ is the scale factor.

$$
y_i^{[l]} = \gamma \hat{z}_i^{[l]} + \beta
$$

Thus, batch normalisation gives you control over what the normalised distribution will look like.

### Convolutions

Convoluations are element-wise products and sums used to scan an image to detect useful features.

Convolutions reduce the size of an image while preserving key features and patterns.

Convolutions allow you to detect key features in different areas of an image using filters that are learnable during training.

#### Padding and Stride

Stride determines how the filter scans the image.

Padding is like a frame on the image. It gives similar importance to the edges and the centre:

Without padding, the corner pixels would only get scanned once during the application of a convolution. However, with **padding**, *each pixel within the image gets visited the same number of times.*

#### Pooling and Upsampling

Pooling reduces the size of the input.

Upsampling increases the size of the input by inferring pixels. There are three types of upsampling:

- Nearest neighbours
- Linear interpolation
- Bi-linear interpolation

Neither pooling nor upsampling have learnable parameters!

#### Transposed Convolutions (a.k.a Deconvolutions)

A transposed convolution uses a learnable filter to upsample an image.

Transposed convolutions thus do have learnable parameters (the filters).

An upsampling layer infers, i.e. calculates, the pixels while transposed convolutions are an upsampling method that uses learnable parameters, unlike upsampling layers.

*Problem:* results have a [checkerboard pattern](http://doi.org/10.23915/distill.00003), because centre pixels are influenced much more, and outer pixels are not.

Upsampling followed by convolution is becoming a more popular technique now to avoid this checkerboard problem.

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford, Metz, and Chintala, 2016)](https://arxiv.org/abs/1511.06434)

## 1.3 Wasserstein GANs with Gradient Penalty

### Mode Collapse

A mode is a peak in the feature distribution.

With real-world datasets, typically there may be multiple modes. For example, in a dataset with 10 different digits, likely there will be 10 different modes.

Mode collapse happens when the generator gets stuck in one mode.

This happens because of BCE loss.

### Problem with BCE Loss

*When the real/fake distributions are far apart, the discriminator does not output useful gradients (feedback) for the generator.*

The objective is to make the generated and real distributions look similar.

Often, however, the discriminator gets better than the generator. When the discriminator improves too much, the function approximated by the BCE loss will contain flat regions.

Flat regions on the cost function cause **vanishing gradients**.

### Earth Mover's Distance (EMD)

The EMD is a function of amount and distance: *The effort to make the generated distribution equal to the real distribution depends on the distance and amount moved.*

It doesn't have flat regions even when the two distributions are very different.

Approximating the EMD solves the BCE's vanishing gradient problems.

### Wasserstein Loss

W-Loss approximates the Earth Mover's Distance:

$$
\min_g \max_c \Epsilon(c(x)) - \Epsilon(c(g(z)))
$$

It looks similar to BCE loss, and helps prevents mode collapse and vanishing gradient problems.

The generator minimises the distance. The generator maximises the distance.

### Condition on Wasserstein Critic

In WGAN-GP, you no longer use a discriminator that classifies fake and real as 0 and 1 but rather a critic that scores images with real numbers.

The critic $c$ needs to be 1-L continuous; that is, the norm of the gradient, the slope of the function, should be at most 1 for every point:

$$
\lVert \nabla f(x) \rVert _2 \leq 1
$$

This condition ensures that W-Loss is validly approximating Earth Mover's Distance.

### 1-Lipschitz Continuity Enforcement

Ways to enforce 1-L Continuity:

- Weight clipping:
  - Force the weights of the critic to a fixed interval
  - Disadvantage: Limits the learning ability of the critic
- Gradient penalty:
  - Regularisation of the critic's gradient

    $$\min_g \max_c \Epsilon(c(x)) - \Epsilon(c(g(z))) + \lambda \Epsilon( \lVert \nabla c(\hat{x}) \rVert _2 - 1 )^2$$

Gradient penalty tends to work better.

It makes the GAN less prone to mode collapse and vanishing gradients; and tries to make the critic be 1-L Continuous, for the loss function to be continuous and differentiable.

[From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)

## 1.4 Conditional GANs & Controllable Generation

### Conditional Generation

| Conditional Generation | Unconditional Generation |
| --- | --- |
| Examples from the classes you want | Examples from random classes |
| Training dataset needs to be labelled | Training dataset doesn't need to be labelled |

Conditional generation requires labelled datasets. Examples can then be generated for the selected class.

Inputs:

- The class is passed to the generator as one-hot vectors.
- The class is passed to the discriminator as one-hot matrices.
- The size of the vector and the number of matrices represent the number of classes.

[Conditional Generative Adversarial Nets (Mirza and Osindero, 2014)](https://arxiv.org/abs/1411.1784)

### Controllable Generation

Controllable generation lets you control the features of the generated *outputs*.

It does not need a labelled training dataset.

The input vector is tweaked in order to get different features on the output.

| Controllable Generation | Conditional Generation |
| --- | --- |
| Examples with the features you want | Examples from the classes you want |
| Training dataset doesn't need to be labelled | Training dataset needs to be labelled |
| Manipulate the z vector | Append a class vector to the input |

Controllable generation is done after training by modifying the z vectors passed to the generator while conditional generation is done during training and requires a labelled dataset.

#### Vector Algebra in the Z-Space

To control output features, you need to find directions in the Z-space.

$$
\text{Original output} = g(v_1)
$$

$$
\text{Controlled output} = g(v_1 + d)
$$

#### Challenges with Controllable Generation

- **Output correlation:** when trying to control one feature, others that are correlated change too.
- **Z-space entanglement** happens when z does not have enough dimensions. It makes controllability difficult, if not impossible (even with low correlation between features).

#### Classifier Gradients

Classifiers can be used to find directions in the Z-space: You can calculate the gradient of the z vectors along certain features through the classifier to find the direction to move the z vectors. *Modify just the noise vector, until the feature emerges.*

#### Disentanglement

Disentangled Z-spaces let you control individual features by corresponding z values directly to them.

Because the components of the noise vectors in the disentangled z-space allow you to change those features that you desire in the output, they're often called latent factors of variation, where the word latent comes from the fact that the information from the noise vectors is not seen directly on the output, but they do determine how that output looks. Sometimes you might hear noise vectors being referred more generally as latents. Then factors of variation means that these are just different factors like hair color and hair length that you want to vary, and only that one factor, that one feature that you are varying when you're varying it, not anything else. Essentially, a disentangled z-space means that there are specific indices on the noise vectors, these specific dimensions that change particular features on the output of your GAN. For instance, if a person on the image generated by a GAN has glasses or not, or whether she has a beard or some features of her hair. Each of these cells will correspond to something that you desire to change, and you can just change the values of that dimension in order to adapt glasses, beard, or hair.

There are supervised and unsupervised methods to achieve disentanglement.

An example of a controllable GAN: [Interpreting the Latent Space of GANs for Semantic Face Editing (Shen, Gu, Tang, and Zhou, 2020)](https://arxiv.org/abs/1907.10786)

## 2.1 Evaluation of GANs

Evaluating GANs is challenging because there is no ground truth: Any discriminator will overfit to the generator it's trained with, and therefore cannot be used to evaluate other generators.

Evaluation metrics try to quantify fidelity and diversity:

- **Fidelity** measures image quality and realism.
- **Diversity** measures variety.

### Comparing Images

**Pixel distance** is simple but unreliable. For example, shifting pixels by one pixel distorts pixel distance even though the image hasn't really changed.

**Feature distance** extracts higher level features of an image, making it more reliable.

### Feature Extraction

Classifiers can be used as feature extractors by *cutting the network at earlier layers.*

*The last pooling layer is most commonly used for feature extraction because it contains the most fine-grained feature information.* Although you could also use previous pooling layers if you want more coarse features.

Fortunately, an extensive number of pre-trained classifiers are available to use.

It is best to use classifiers that have been trained on large datasets, e.g. ImageNet!

#### Inception-v3 and Embeddings

A commonly used feature extractor is an Inception-v3 classifier pre-trained on ImageNet, with the output layer cut off.

These features are called embeddings: *Extracted features are frequently called an embedding because they’re condensed into a lower-dimensional space.*

The embeddings are compared to get the feature distance.

### [Fréchet Inception Distance (FID)](https://nealjean.com/ml/frechet-inception-distance/)

FID calculates the distance between reals and fakes using the Inception model and multivariate normal Fréchet distance.

Univariate Normal Fréchet Distance:

$$
d(X, Y) = (\mu_X - \mu_Y)^2 + (\sigma^2_X - \sigma^2_Y - 2 \sigma_X \sigma_Y) = (\mu_X - \mu_Y)^2 + (\sigma_X - \sigma_Y)^2
$$

Multivariate Normal Fréchet Distance:

$$
d(X, Y) = \lVert \mu_X - \mu_Y \rVert^2 + Tr( \sigma_X + \sigma_Y - 2 \sqrt{\sigma_X \sigma_Y} )
$$

$Tr$ is the trace, the sum of the diagonal elements; i.e., in this case the sum of the variance of each of the distribution.

Real and fake embeddings are two multivariate normal distributions. FID looks at the mean and the covariance matrices of the real and fake multivariate normal distributions and calculates how far apart those statistics are from each other. Therefore, the lower the FID, the closer the distributions.

Use a large sample size to reduce noise!

Shortcomings of FID:

- Uses pre-trained Inception model, which may not capture all features.
- Needs a large sample size.
- Slow to run.
- Limited statistics used: only mean and covariance.

### [Inception Score (IS)](https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)

The Inception Score (IS) is another evaluation metric used. It tries to capture fidelity and diversity:

$$
IS = \exp(\Epsilon_{x \sim p_{\epsilon}} D_{KL}(p(y|x) \Vert p(y)))
$$

where $D_{KL}$ is the KL Divergence:

$$
D_{KL}(p(y|x) \Vert p(y)) = p(y|x) \log(\frac{p(y|x)}{p(y)})
$$

- Conditional distribution $p(y|x) =$ fidelity (low entropy)
- Marginal distribution $p(y) =$ diversity (high entropy)

[It is worse than FDI](https://arxiv.org/abs/1801.01973) as it has many shortcomings:

- Can be exploited or gamed too easily, e.g. generating one realistic image of each class
- Only looks at fake images, makes no comparison with real images
- Can miss useful features, ImageNet doesn't teach a model all features

### Sampling and Truncation

Sample fakes using the training or prior distribution of z

**Truncation trick:** Sample at test time from a normal distribution with its tails clipped. Truncate more for higher fidelity, lower diversity:

- If you want higher fidelity, you want to sample around 0 and truncate a larger part of the tails.
- If you want greater diversity, then you want to sample more from the tails of the distribution and have a lower truncation value.

Human evaluation is still necessary for sampling.

[**HYPE:** A Benchmark for Human eYe Perceptual Evaluation of Generative Models](https://arxiv.org/abs/1904.01121)

### Precision and Recall

Precision

- relates to fidelity: you can see how high the quality of the generator's images is
- looks at overlap between reals and fakes, over how much extra gunk the generator produces

Recall

- relates to diversity: you can see if the generator models all the reals or not
- looks at overlap between reals and fakes, over all the reals that the generator cannot model

[Improved Precision and Recall Metric for Assessing Generative Models (Kynkäänniemi, Karras, Laine, Lehtinen, and Aila, 2019)](https://arxiv.org/abs/1904.06991)

Precision is to fidelity as recall is to diversity.

Models tend to be better at recall.

Use **truncation trick** to improve precision!

## 2.2 Disadvantages and Biases

GANs have *amazing results* but shortcomings as well.

A significant advantage with GANs is that they can produce high quality realistic results. To the human eye, you could be fooled into believing these people actually exist. However, a downside is that during training, the model can be unstable and take considerable time to train.

Advantages of GANs:

- Amazing empirical results - especially with fidelity (i.e. precision, quality)
- Fast inference (image generation during testing)

Disadvantages of GANs:

- Lack of intrinsic evaluation metrics
- Unstable training
- No density estimation
- Inverting is not straightforward

### Alternative Generative AI Models

A GAN takes noise as input and never directly sees the real image.

VAEs work with two models, an encoder and a decoder, that take a real image, find a good way of representing that image in latent space, and then reconstruct a realistic image.

VAEs minimise divergence between generated and real distributions - which is an easier optimisation technique resulting in more stable training but lower-fidelity / blurrier results.

What is the same is that the model tries to model $P(features|class)$ given some noise: *A generative model can be any machine learning model that tries to model* $P(X|Y)$.

- Variational Autoencoders:
  - Density estimation
  - Invertible
  - Stable training
  - At the expense of lower-fidelity / lower-precision / lower-quality results
- Autoregressive Models:
  - Supervised model
  - Rely on previous pixels to generate next pixel (similar idea to RNNs)
- Flow Models:
  - Use invertible mappings
- Hybrid Models
- [Score-Based Generative Models](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_(Optional_Notebook)_Score_Based_Generative_Modeling.ipynb)

### [Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)

Machine learning bias has a disproportionately negative effect on historically underserved populations.

Proprietary risk assessment software is difficult to validate, and misses import considerations about people.

### Ways Bias is Introduced

Bias can be introduced into a model at each step of the process.

- Training Bias:
  - *No variation* in who or what is represented in training data
  - Bias in *collection methods*
  - *Diversity* of the data labellers
- Evaluation Bias:
  - Images can be biased to reflect "correctness" in the dominant culture
- Model Architecture Bias:
  - Can be influenced by the coders who designed the architecture or optimised the code

## 2.3 [StyleGANs and Advancements](https://medium.com/@jonathan_hui/gan-stylegan-stylegan2-479bdf256299)

GANs have improved because of

- Stability: longer training and better images
  - Use standard deviation in batch to encourage diversity
  - Enforce 1-Lipschitz continuity, e.g. WGAN-GP and Spectral Normalisation
  - Take average of generator weights across several iterations $\bar{\theta} = \frac{1}{n} \sum_{i=0}^n{\theta_i}$ which gives much smoother results
  - Use **progressive growing** which gradually trains GANs at increasing resolutions
- Capacity: improved hardware has enabled larger models which can use higher resolution images
- Diversity: increasing variety in generated images

### StyleGANs

*Style* is any variation in the image.

StyleGAN goals:

1. Greater *fidelity* on high-resolution images
2. Increased *diversity* of outputs
3. More *control* over image features

Main components of StyleGANs:

- Progressive growing
- Noise mapping network
- Adaptive instance normalisation (AdaIN)
- Style Mixing
- Stochastic Noise

### Progressive Growing

Progressive growing gradually doubles image resolution used for generator training.

This helps with faster, more stable training for higher resolutions.

### Noise Mapping Network

Noise mapping allows for a more disentangled noise space, i.e. mapping onto single output features = more control of image features.

The intermediate noise vector $w$ is used as an input to the generator.

### Adaptive Instance Normalisation (AdaIN)

The purpose of AdaIN layers is to *transfer style information from the intermediate noise vector* $w$ *onto the generated image.*

They also renormalize the statistics so each block overrides the one that came before it.

$$
AdaIN(x_i, y) = y_{s, i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b, i}
$$

1. Step: Instance normalisation $\frac{x - \mu(x)}{\sigma(x)}$
2. Step: Adaptive styles $y_{s,i}, y_{b,i}$

### Style Mixing and Stochastic Noise

Stochastic noise causes small variations to output (e.g. hair strands, wrinkles, etc.).

Style mixing increases diversity that the model sees during training.

Coarse or fineness depends on where in the network style or noise is added:

- Earlier for coarser variation
- Later for finer variation

[A Style-Based Generator Architecture for Generative Adversarial Networks (Karras, Laine, and Aila, 2019)](https://arxiv.org/abs/1812.04948)

## 3.1 GANs for Data Augmentation and Privacy

### GAN Applications

- Image-to-image translation
- Image editing, art, and video
- Data augmentation
- Other fields use adversarial techniques for realism and robustness.

### Data Augmentation

Supplement data when real data is too expensive or rare.

GANs are well suited for this.

Use GANs to generate fake data when real data are too scarce!

[RandAugment: Practical automated data augmentation with a reduced search space (Cubuk, Zoph, Shlens, and Le, 2019)](https://arxiv.org/abs/1909.13719)

#### Pros and Cons

Pros:

- Can be better than hand-crafted synthetic examples
- Can generate more labelled examples
- Can improve a downstream model's generalisation

Cons:

- Can be limited by the available data in diversity
- Can overfit to the real training data

### GANs for Privacy

GANs can be useful for preserving privacy, e.g. in the case of sensitive medical data.

Although generated samples may mimic the real ones too closely. *Post-processing may help avoid this data leakage.*

### GANs for Anonymity

GANs can enable healthy anonymous expression for stigmatised groups.

However, GANs for anonymisation can be used for good just as for evil - e.g. identity theft, deepfakes.

### [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data (Such et al. 2019)](https://arxiv.org/abs/1912.07768)

Essentially, a [GTN](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W1_Generative_Teaching_Networks_(Optional).ipynb) is composed of a generator (i.e. teacher), which produces synthetic data, and a student, which is trained on this data for some task. The key difference between GTNs and GANs is that GTN models work cooperatively (as opposed to adversarially).

[Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (Zakharov, Shysheya, Burkov, and Lempitsky, 2019)](https://arxiv.org/abs/1905.08233)

[De-identification without losing faces (Li and Lyu, 2019)](https://arxiv.org/abs/1902.04202)

## 3.2 Image-to-Image Translation with Pix2Pix

### Image-to-Image Translation

Image-to-image translation transforms images into different styles.

GANs' realistic generation abilities are well-suited to image-to-image translation tasks.

**Paired translation** means that you have input-output pairs that map exactly onto each other (1-to-1).

Other types of translation include test-to-image or image-to-video.

### Pix2Pix

Pix2Pix inputs and outputs are similar to a conditional GAN:

- Take in the original image, instead of the class vector
- No explicit noise as input

### PatchGAN

The PatchGAN discriminator outputs a matrix of values, each between $0$ and $1$ - representing feedback on the realness within each region or patch of the image.

### U-Net

Pix2Pix's generator is a U-Net, a very successful architecture for image segmentation.

U-Net is an encoder-decoder, with same-size inputs and outputs.

U-Net uses skip connections, which improve gradient flow to the encoder, and thus help the decoder learn details from the encoder directly.

Dropout in some decoder blocks adds noise to the network.

### Pixel Distance Loss Term

Pix2Pix adds a Pixel Distance Loss term to the generator loss function.

This loss term calculates the difference between the fake and the real target outputs.

$$
\text{BCE Loss} + \lambda \sum_{i=1}^n{ \lvert \text{generated output} - \text{real output} \rvert }
$$

It softly encourages the generator with this additional supervision:

- *The real output image and the generated image should be encouraged to be similar.*
- The target output labels are the supervision.
- The generator essentially "sees" these labels.

### Pix2Pix Architecture

- U-Net generator
- PatchGAN discriminator
  - Inputs input image and paired output (either real target or fake)
  - Outputs classification matrix
- Generator loss has a regularisation term

### Advancements

Pix2PixHD and GauGAN are successors of Pix2Pix, designed for higher resolution images. They highlight opportunities for image editing using paired image-to-image translation.

[Image-to-Image Translation with Conditional Adversarial Networks (Isola, Zhu, Zhou, and Efros, 2018)](https://arxiv.org/abs/1611.07004)

[Pix2PixHD](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W2_Pix2PixHD_(Optional).ipynb) which synthesizes high-resolution images from semantic label maps. Proposed in [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (Wang et al. 2018)](https://arxiv.org/abs/1711.11585), Pix2PixHD improves upon Pix2Pix via multiscale architecture, improved adversarial loss, and instance maps.

[Super-Resolution GAN (SRGAN)](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W2_SRGAN_(Optional).ipynb), a GAN that enhances the resolution of images by 4x, proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (Ledig et al. 2017)](https://arxiv.org/abs/1609.04802).

[Patch-Based Image Inpainting with Generative Adversarial Networks (Demir and Unal, 2018)](https://arxiv.org/abs/1803.07422)

[GauGAN](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W2_GauGAN_(Optional).ipynb), which synthesizes high-resolution images from semantic label maps, which you implement and train. GauGAN is based around a special denormalization technique proposed in [Semantic Image Synthesis with Spatially-Adaptive Normalization (Park et al. 2019)](https://arxiv.org/abs/1903.07291)

## 3.3 Unpaired Translation with CycleGAN

Unlike paired image-to-image translation (e.g. Pix2Pix), unpaired image-to-image translation is unsupervised - there is no longer a clear output target.

Unpaired image-to-image translation

- learns a mapping between two piles of images
- examines common elements of the two piles (content) and unique elements of each pile (style)

### CycleGAN: Two GANs

CycleGAN has four components - two generators and two discriminators:

- The discriminators are PatchGANs.
- The generators are similar to a U-Net and DCGAN generator with additional skip connections.

The inputs to the generators and discriminators are similar to Pix2Pix, except:

- there are no real target outputs
- each discriminator is in charge of one pile of images

### Cycle Consistency

Cycle consistency helps transfer uncommon style elements between the two GANs, while maintaining common content.

*When styles between two piles are transferred, the original content can be recovered.*

Add an extra **Cycle Consistency Loss** term to each generator to softly encourage cycle consistency. Without this extra loss term, outputs show signs of mode collapse.

Cycle consistency is used in both directions.

### Least Squares Loss

Like BCE Loss, Least Squares Loss also calculates how much you stray from the ground truth:

| Discriminator Least Squares Loss | Generator Least Squares Loss |
| --- | --- |
| $\Epsilon_x[(D(x) - 1)^2] + \Epsilon_z[(D(G(z)) - 0)^2]$ | $\Epsilon_z[(D(G(z)) - 1)^2]$ |

Least Squares Loss reduces vanishing gradients and helps with mode collapse.

### Identity Loss

Identity loss is *optionally* added to help with colour preservation.

### CycleGAN Architecture

CycleGAN is composed of two GANs.

Generators have 6 loss terms in total, 3 each of:

- Least Squares Adversarial Loss
- Cycle Consistency Loss
- Identity Loss

Discriminator is simpler, with Least Squares Adversarial Loss using PatchGAN.

### Applications & Variants

- Democratised art and style transfer
- Medical data augmentation
- Creating paired data

UNIT and MUNIT are other models for unpaired (unsupervised) image-to-image translation.
