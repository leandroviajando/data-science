# Convolutional Neural Networks

## Introduction

Convolutions involve sliding a filter throughout the whole input volume. Thus, **convolutions allow a feature detector to be used in multiple locations throughout the whole input volume.**

For example, a `3 x 3` filter for vertical edge detection:

```python
[[1, 0, -1],
[1, 0, -1],
[1, 0, -1]]
```

A Sobel filter puts a little bit more weight on the central row,
the central pixel, and this makes it maybe a little bit more robust:

```python
[[1, 0, -1],
[2, 0, -2],
[1, 0, -1]]
```

Or similarly one could put more weight on the centre of the image:

```python
[[0, 1, -1, 0],
[1, 3, -3, -1],
[1, 3, -3, -1],
[0, 1, -1, 0]]
```

By letting all of these numbers be parameters and learning them automatically from data, we find that neural networks can actually learn low level features, features such as edges, even more robustly than computer vision researchers are generally able to code up these things by hand.

### Convolutions

A fully-connected layer with 100 neurons, given a $256 \times 256$ RGB image as input, has $(256 \times 256 \times 3 + 1) \times 100 = 19,660,900$ parameters.

A convolutional layer with 128 filters of size $7 \times 7$ has only $(7 \times 7 \times 3 + 1) \times 128 = 18,944$ parameters.

**Convolutional layers provide sparsity of connections:** The next layer will depend only on a small number of activations from the previous layer.

Weight sharing reduces significantly the number of parameters in a neural network, and sparsity of connections allows us to use a smaller number of inputs thus reducing even further the number of parameters.

```python
def conv_single_step(a_slice_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = a_slice_prev * W
    # Sum over all entries of the volume s.
    Z = s.sum()
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z += float(b)

    return Z
```

### Padding and Strided Convolutions

The main benefits of padding are:

- It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as you go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer.
- It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels at the edges of an image.

```python
def zero_pad(X: np.ndarray, pad: int) -> np.ndarray:
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=(0, 0))

    return X_pad
```

The output volume will be

$$
n_H^l = \lfloor \frac{n_H^{[l-1]} - f + 2p}{s} \rfloor + 1
$$

$$
n_W^l = \lfloor \frac{n_W^{[l-1]} - f + 2p}{s} \rfloor + 1
$$

In the forward pass, you will take many filters and convolve them on the input. Each "convolution" gives you a 2D matrix output. You will then stack these outputs to get a 3D volume:

$$
n_C = \text{number of filters}
$$

Types of padded convolutions:

- **Valid** convolution (no padding): $p=0$
- **Same** convolution (padding): $p=\frac{f-1}{2}$

$f$ is usually odd.

### Convolutional Network Layers

- Convolution
- Pooling: pool if feature filtered in convolution found in that part of matrix
  - Hyperparameters:
    - Stride, usually $s = f$
    - No padding (i.e. $p=0$ in above size calculations for $n_W, n_H, n_C$)
    - Type: `max` or `avg`
- Fully Connected

For pooling, $f = s = 2$ is a common choice.

A typical structure of a ConvNet consists of multiple CONV layers, followed by a POOL layer to flatten the volume, and FC layers in the last few layers to generate the output.

## Deep Convolutional Models

**Transfer learning:** It turns out that a neural network architecture that works well on one computer vision tasks often works well on other tasks as well.

### LeNet-5

| Layer | Dimensions |
| --- | --- |
| Inputs: grey-scale images | $32 \times 32 \times 1$ |
| Filter | $5 \times 5$ with stride $s=1$ |
| Hidden Layer | $28 \times 28 \times 6$ |
| Avg Pool | $f=2, s=2$ |
| Hidden Layer | $14 \times 14 \times 6$ |
| Filter | $5 \times 5, s=1$ |
| Hidden Layer | $10 \times 10 \times 16$ |
| Avg Pool | $f=2, s=2$ |
| Hidden Layer | $5 \times 5 \times 16$ |
| FC Layer | $120$ |
| FC Layer | $84$ |
| Ouput | $\hat{y}$ |

Note height and width $n_H, n_W \uarr$, and the number of channels $n_C \darr$.

Note also no padding was used - which was not used yet in 1998.

### AlexNet

Similar to LeNet but uses Max Pools, ReLU, softmax output activation and is much bigger: $\approx 60m$ parameters.

### VGG-16

Focuses on always having Convolutional Layers with $3 \times 3$ filters and stride $s=1$; and Max Pools of $2 \times 2$ and stride $s=2$.

Height and width $n_H, n_W \darr$, and the number of channels $n_C \uarr$.

### ResNet

Very, very deep neural networks are difficult to train, because of vanishing and exploding gradient: during gradient descent, as you backpropagate from the final layer back to the first layer, you are multiplying by the weight matrix on each step, and thus the gradient can decrease exponentially quickly to zero (or, in rare cases, grow exponentially quickly and "explode," from gaining very large values).

**Skip connections** allow you to take the activation from one layer and feed it to another layer much deeper in the neural network.

$$
\begin{align}
a^{[l+2]} &= g(z^{[l+2]} + a^{[l]}) \\
 &= g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + a^{[l]}) \\
 &= g(W^{[l+2]} g(W^{[l+1]}a^{[l]} + b^{[l+1]}) + b^{[l+2]} + a^{[l]}) \\
\end{align}
$$

The skip connection makes it easy for the network to learn an identity mapping between the input and the output within the ResNet block.

The skip connection helps the gradient to backpropagate and thus helps train deeper networks.

Thus, ResNet enables you to train very, very deep networks, sometimes even networks of over 100 layers.

Then, if using L2 regularization / weight decay, the weights (and for sake of argument, let's say also the bias) will go towards zero, and $a^{[l+2]} = g(a^{[l]})$, which is just $a^{[l+2]} = a^{[l]}$ if using ReLU activations since $a \geq 0$.

$$
a^{[l+2]} = a^{[l]}
$$

What goes wrong in very deep networks without this residual of the skip connections is that when you make the network deeper and deeper, it's actually very difficult for it to choose parameters that learn even the identity function which is why a lot of layers end up making your result worse rather than making your result better.

The main reason the residual network works is that it's so easy for these extra layers to learn the identity function that you're kind of guaranteed that it doesn't hurt performance and then a lot the time you maybe get lucky that it even helps performance. At least it is easier to go from a decent baseline of not hurting performance and then gradient descent can only improve the solution from there.

#### The Identity Block

Note that for this to work ResNets tend to use convolutional layers of the **same size** / dimensions.

```python
def identity_block(X: tf.Tensor, *, f: int, filters: List[int], training: bool = True, initializer=random_uniform) -> tf.Tensor:
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # channels axis
    X = Activation("relu")(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # channels axis
    X = Activation("relu")(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # channels axis

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X
```

#### The Convolutional Block

Otherwise, use a matrix $W_s$ (whose weights can be learned) to get same dimensions:

$$
\begin{align}
a^{[l+2]} &= g(z^{[l+2]} + W_s a^{[l]}) \\
 &= g(w^{[l+2]}a^{[l+1]} + b^{[l+2]} + W_s a^{[l]}) \\
\end{align}
$$

```python
def convolutional_block(X: tf.Tensor, *, f: int, filters: List[int], s: int = 2, training: bool = True, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                   also called Xavier uniform initializer.

    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, kernel_size=(1, 1), strides=(s, s), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # channels axis
    X = Activation("relu")(X)

    X = Conv2D(F2,  kernel_size=(f, f), strides=(1, 1), padding="same", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # channels axis
    X = Activation("relu")(X)

    X = Conv2D(F3, kernel_size=(1, 1), strides=(1, 1), padding="valid", kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)  # channels axis

    X_shortcut = Conv2D(F3, kernel_size=(1, 1), strides=(s, s), padding="valid", kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)  # channels axis

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X
```

Very deep residual networks are built by stacking these blocks together:

```python
def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)
    X = ZeroPadding2D(padding=(3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256])
    X = identity_block(X, f=3, filters=[64, 64, 256])

    ## Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512])
    X = identity_block(X, f=3, filters=[128, 128, 512])
    X = identity_block(X, f=3, filters=[128, 128, 512])

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024])
    X = identity_block(X, f=3, filters=[256, 256, 1024])
    X = identity_block(X, f=3, filters=[256, 256, 1024])
    X = identity_block(X, f=3, filters=[256, 256, 1024])
    X = identity_block(X, f=3, filters=[256, 256, 1024])

    # Stage
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048])
    X = identity_block(X, f=3, filters=[512, 512, 2048])

    # AVGPOOL
    X = AveragePooling2D()(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", kernel_initializer = glorot_uniform(seed=0))(X)

    return Model(inputs=X_input, outputs=X)
```

### Inception

Inception networks apply a number of different convolutions, e.g. $3 \times 3, 5 \times 5$, Max Pooling. This is computationally expensive.

It turns out that *first applying a $1 \times 1$ filter shrinks the number of channels* resulting in a $\approx 10$ times lower computational cost.

Note how

- a $1 \times 1$ convolutional layer with a small number of filters reduces $n_C$
- a 2D pooling layer on the other hand reduces $n_W, n_H$

Making an inception network deeper (by stacking more inception blocks together) can improve performance, but can also lead to overfitting and increase in computational cost.

### MobileNet

Most neural networks are quite computationally expensive. If you want your neural network to run on a device with less powerful CPU or a GPU at deployment, then there's another neural network architecture called the MobileNet that is optimised to run on mobile and other low-power applications.

*It is very efficient for object detection and image segmentation tasks.* Its architecture has three defining characteristics:

- Depthwise separable convolutions: for lightweight feature filtering and creation, dealing with spatial and depth (number of channels) dimensions
- Thin input and output bottlenecks between layers: preserving important information on either end of the block
- Shortcut connections between bottleneck layers

A **depthwise separable convolution** performs a combined depthwise and a pointwise convolution instead of a normal convolution.

- The depthwise convolution convolves each channel in the input volume with a separate filter.
- The pointwise convolution convolves the output volume with $1 \times 1$ filters.

Note the result may or may not have the same number of channels as the input.

The use of this so-called "bottleneck" reduces the depth of the volume and thus helps reduce the computational cost of applying other convolutional layers with different filter sizes.

The use of bottlenecks does not seem to hurt the performance of the network.

### EfficientNet

This implementation allows scaling up or down the *resoluiton, depth and width* of a network depending on the device.

### Transfer Learning

To adapt a classifier to new data, drop the top layer, add a new classification layer, and train only on that layer.

```python
def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''

    input_shape = image_shape + (3,)

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,  # drop last layer, for transfer learning
                                                   weights="imagenet") # from imageNet

    # freeze the base model by making it non-trainable
    base_model.trainable = False

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape)

    # apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)

    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tfl.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tfl.Dropout(.2)(x)

    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tfl.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    return model
```

Fine-tune the final layers of your model to capture high-level details near the end of the network and potentially improve accuracy.

```python
base_model = model2.layers[4]
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 120

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Define a BinaryCrossentropy loss function. Use from_logits=True
loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# Define an Adam optimizer with a learning rate of 0.1 * base_learning_rate
optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate * .1)
# Use accuracy as evaluation metric
metrics = ["accuracy"]

model2.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=metrics)

fine_tune_epochs = 5
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model2.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
```

### Data Augmentation

- Mirroring (left-right)
- Random cropping
- Colour shifting (adding distortions to the RGB channels)
  - PCA colour augmentation

```python
def data_augmenter():
    """
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    """
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"))
    data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomRotation(.2))

    return data_augmentation
```

### Ensembling

Train a few different convolutional models independently, and average their outputs.

## Object Localisation and Detection

### Object Localisation

Target label $y = [p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3, ...]$ where $p_c$ is a binary value, whether an object was detected or not; $b_x, b_y, b_h, b_w$ define the "box" where in the image the object is located; and $c_i$ is $1$ for the class of object detected.

#### Landmark Detection

When building a neural network that inputs a picture of a person's face and outputs $N$ landmarks on the face (assume that the input image contains exactly one face), we need two coordinates for each landmark, thus we need $2N$ output units.

### Object Detection

*Sliding Windows Detection*, using smaller, larger windows, and so on

#### Convolutional Implementation of Sliding Windows

TODO: code

#### YOLO (You Only Look Once) Algorithm

##### Bounding Box Predictions

The YOLO algorithm splits an image into a grid, e.g. $3 \times 3$ cells.

If an object appears across more than one grid cell, it is assigned to the grid cell where its midpoint is located.

##### Intersection Over Union

*Map localisation to accuracy:* IOU is a measure of the overlap between two bounding boxes - the prediction boundary box and the target boundary box:

$$
IOU = \frac{\text{size of intersection}}{\text{size of union}}
$$

Prediction "correct" if $IOU \geq .5$

##### Non-Max Suppression

Multiple grid cells will raise their hand that they've detected the object.

Non-max suppression will highlight the one with the highest probability, and then suppresses all boxes that have high IOU with the highlighted one - thus removing duplicates:

Given an output prediction,

1. discard all boxes with $p_c \leq .6$ (i.e. a threshold for low-probability predictions)
2. pick the box with the largest $p_c$ and output it as the prediction
3. discard any remaining boxes with $IOU \geq .5$ with the box output in the previous step; and continue from step 2 as long as there are predictions left

##### Anchor Boxes

When trying to detect multiple objects, there may be more than one object in a grid cell.

Anchor boxes encode the shape of an object, thus enabling to differentiate between objects.

*Each object in the training image is assigned to the grid cell that contains the object's midpoint, AND to the anchor box for the grid cell with the highest $IOU$.*

Putting it all together, for example, if you choose a $3 \times 3$ grid, 2 anchors and 3 classes, then the size of the output $y = [p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3, ...]$ grows to have size

$$
3 \times 3 \times 2 \times (5 + 3)
$$

where $5+3$ was the initial size of the output, as defined above, but it's now multiplied by the number of grid cells and anchor boxes.

#### Semantic Image Segmentation

Binary semantic segmentation: Assign labels $0$ or $1$ to every pixel in the image.

Multi-class semantic segmentation is also possible.

#### Transpose Convolutions

A transpose convolution applies a filter that is larger than the input size to get a larger output, e.g. $(2 \times 2) * (3 \times 3) \rarr (4 \times 4)$.

#### U-Net

U-Net combines transpose convolutions with skip connections. This allows it to combine both the lower resolution, but high level,
spatial, high level contextual information, as well as the low level, but more detailed texture-like information in order to make a decision as to whether a certain pixel is part of a cat or not.

*It uses an equal number of convolutional blocks and transposed convolutions for downsampling and upsampling. Skip connections are used to prevent border pixel information loss and overfitting.*

When using the U-Net architecture with an input $h \times w \times c$, where $c$ denotes the number of channels, the output has shape $h \times w \times k$.

## Face Recognition

- **Face verification** requires comparing a new picture against one person’s face, i.e. a $1:$ matching problem: "Is this the acclaimed person?"
- **Face recognition** requires comparing a new picture against $K$ persons’ faces, i.e. a $1:K$ matching problem: "Who is this person?"

Triplet loss is an effective loss function for training a neural network to learn an encoding of a face image.

The same encoding can be used for verification and recognition. Measuring distances between two images' encodings allows you to determine whether they are pictures of the same person.

Ways to improve your facial recognition model:

- Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. Then, given a new image, compare the new face to multiple pictures of the person. This would increase accuracy.
- Crop the images to contain just the face, and less of the "border" region around the face. This preprocessing removes some of the irrelevant pixels around the face, and also makes the algorithm more robust.

### One-Shot Learning

One-shot learning problem: Need to recognise a person given just one image of their faces.

Solution: Learn a "similarity" function outputting the degree of difference between to images $d(img_1, img_2)$.

The job of the function $d$ is to input two faces and tell you how similar or how different they are. A good way to do this is to use a Siamese network.

### Siamese Networks

The idea of running two identical, convolutional neural networks on two different inputs and then comparing them is called a Siamese neural network architecture.

$$
d(x^{(1)}, x^{(2)}) = \lVert f(x^{(1)}) - f(x^{(2)}) \rVert ^2
$$

The parameters of the neural network define an encoding $f(x^{(i)})$ s.t. $d$ will be small if both are the same person and large otherwise.

### Triplet Loss

Consider an anchor $A$, and a positive example $P$ (i.e. same person) and a negative example $N$ (i.e. different person). Thus the objective is:

$$
d(A, P) = \lVert f(A) - f(P) \rVert ^2 \leq \lVert f(A) - f(N) \rVert ^2 = d(A, N)
$$

$$
\dArr
$$

$$
\lVert f(A) - f(P) \rVert ^2 - \lVert f(A) - f(N) \rVert ^2 + \alpha \leq 0
$$

where $\alpha \gt 0$ is the margin parameter (similar to a large-margin classifier, SVM).

This gives the Triplet Loss and Cost Functions

$$
L(A, P, N) = \max(\lVert f(A) - f(P) \rVert ^2 - \lVert f(A) - f(N) \rVert ^2 + \alpha, 0)
$$

$$
J = \sum_{i=1}^m{L(A^{(i)}, P^{(i)}, N^{(i)})}
$$

If $A, P, N$ are chosen randomly, the above condition will be satisfied quite easily. So want to specifically select triplets that are "hard", i.e. $d(A, P) \approx d(A, N)$, to train on.

```python
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

    Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
    """

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above.
    encoding = img_to_encoding(image_path, model)
    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(encoding - database[identity])
    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

def recognise(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.

    Arguments:
        image_path -- path to an image
        database -- database containing image encodings along with the name of the person on the image
        model -- your Inception model instance in Keras

    Returns:
        min_dist -- the minimum distance between image_path encoding and the encodings from the database
        identity -- string, the name prediction for the person on image_path
    """

    # Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
    encoding =  img_to_encoding(image_path, model)

    # Step 2: Find the closest encoding

    # Initialize "min_dist" to a large value, say 100
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current db_enc from the database.
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity
```

### Face Verification and Binary Classification

Alternatively, face verification can also be posed as a binary classification problem.

$$
\hat{y} = \sigma(\sum_{k=1}^{128}{w_k |f(x^{(i)})_k - f(x^{(j)})_k| + b})
$$

## Neural Style Transfer

In neural style transfer, we train the pixels of an image, and not the parameters of a network.

Consider an image $C$ which provides the content, an image $S$ which defines the style, and an image $G$ which has been generated given the provided content and style.

1. Initiate $G$ randomly
2. Use gradient descent to minimise the cost:

$$
J(G) = \alpha J_{\text{content}}(C, G) + \beta J_{\text{style}}(S, G)
$$

### Content Cost Function

If layer $l$ activations $a^{[l](C)}, a^{[l](G)}$ are similar, both images have similar content:

$$
J_{\text{content}}(C, G) = \frac{1}{2} \lVert a^{[l](C)} - a^{[l](G)} \rVert ^2
$$

```python
def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    a_C = content_output[-1]
    a_G = generated_output[-1]

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_C]))

    # Compute the cost with tensorflow
    J_content =  (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content
```

- The content cost takes a hidden layer activation of the neural network, and measures how different $a(C)$ and $a(G)$ are.
- When you minimize the content cost later, this will help make sure $G$ has similar content as $C$.

### Style Cost Function

Define style as the *correlation between activations across channels $k$* - calculated as Gram matrices:

$$
G_{kk'}^{[l](S)} = \sum_{i=1}^{n_H}{ \sum_{j=1}^{n_W}{ a_{i,j,k}^{[l](S)} a_{i,j,k'}^{[l](S)} } }
$$

$$
G_{kk'}^{[l](G)} = \sum_{i=1}^{n_H}{ \sum_{j=1}^{n_W}{ a_{i,j,k}^{[l](G)} a_{i,j,k'}^{[l](G)} } }
$$

$$
J_{\text{style}}^{[l]}(S, G) = \frac{1}{(2 n_H^{[l]} n_W^{[l]} n_C^{[l]})^2} \sum_k \sum_{k'}(G_{kk'}^{[l](S)} - G_{kk'}^{[l](G)})^2
$$

$$
J_{\text{style}}(S, G) = \sum_l \lambda^{[l]} J_{\text{style}}^{[l]}(S, G)
$$

```python
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A, A, transpose_b=True)

    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = (1 / (4 * n_C**2 * (n_H * n_W)**2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style
```

- The style of an image can be represented using the Gram matrix of a hidden layer's activations.
- You get even better results by combining this representation from multiple different layers.
- This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
- Minimizing the style cost will cause the image $G$ to follow the style of the image $S$.

The total cost is a linear combination of the content cost $J_{\text{content}}(C, G)$ and the style cost $J_{\text{content}}(C, G)$. $\alpha$ and $\beta$ are hyperparameters that control the relative weighting between content and style.

```python
@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style

    return J

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C

        a_G = vgg_model_outputs(generated_image)

        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G)

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style)

    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J
```

- Neural Style Transfer is an algorithm that, given a content image $C$ and a style image $S$, can generate an artistic image
- It uses representations (hidden layer activations) based on a pretrained ConvNet.
- The content cost function is computed using one hidden layer's activations.
- The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
- Optimizing the total cost function results in synthesizing new images.

## 1D and 3D Generalizations

Convolutions can similarly be applied to 1D and 3D data.
