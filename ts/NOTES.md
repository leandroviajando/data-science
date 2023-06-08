# [Sequences, Time Series and Prediction](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction)

[Deep Learning.AI TensorFlow Developer Public Repo](https://github.com/https-deeplearning-ai/tensorflow-1-public)

A combination of neural network architectures (DNNs, Convolutions, RNNs / LSTMs) are often best for predicting time series.

The MAE is a good metric for measuring accuracy of predictions for time series because it doesn't heavily punish larger errors like square errors do.

## 1. Sequences and Prediction

Use cases:

- Forecasting
- Imputation:
  - Backtracking or projecting, i.e. filling up unknown (usually past or missing) data points

Examples:

- **Univariate** time series, e.g. hour by hour temperature
- **Multivariate** time series, e.g. hour by hour weather

### Common patterns in Time Series

Most time series data will contain elements of

- **Trends:** an overall direction for data, regardless of direction
- **Seasonality:** patterns repeating in predictable intervals / regular changes in the shape of the data
- **Autocorrelation:** correlated with a delayed copy of itself, often called a lag; i.e., a predictable shape, even if the scale is different
- **Noise:** unpredictable changes in time series data

If the time series' behaviour does not change over time it is **stationary**.

If it does change over time, i.e. a *disruptive event breaking a trend or seasonality*, e.g. an initial upward trend followed by a downward trend, is **non-stationary**.

### Train, Validation and Test Sets

We typically want to split the time series into a training period, a validation period and a test period. This is called **fixed partitioning**. If the time series has some seasonality, you generally want to ensure that each period contains a whole number of seasons.

At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period. This is called **roll-forward partitioning**. You could see it as doing fixed partitioning a number of times, and then continually refining the model as such.

The test period may also be chosen to be simply future data.

### Metrics for evaluating performance

```python
errors = forecasts - actual

mse = np.square(errors).mean()  # most common

rmse = np.sqrt(mse)  # transform mse to same scale as original 'errors'

mae = np.abs(errors).mean()  # does not penalise large errors as much

mape = np.abs(errors / x_valid).mean()  # gives an idea of the size of the errors compared to the values
```

### Forecasting: Moving Average and Differencing

A **moving average** is calculated over an averaging window, e.g. 30 days, and eliminates a lot of the noise, i.e. roughly emulating the original series without anticipating trend or seasonality.

### Trailing versus Centred Windows

Note that we used the trailing window when computing the moving average of present values But we used a centered window to compute the moving average of past values. Moving averages using centered windows can be more accurate than using trailing windows. But we can't use centered windows to smooth present values since we don't know future values. However, to smooth past values we can afford to use centered windows.

## 2. Deep Neural Networks for Time Series

### Preparing features and labels

Our feature is effectively a number of values in the series, with our label being the next value. We'll call that number of values that will treat as our feature, the **window size**, where we're taking a window of the data and training an ML model to predict the next value.

A **windowed dataset** is thus a fixed-size subset of a time-series. Note `drop_remainder=True` ensures that all rows in the data window are the same length by cropping data.

```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)

for x, y in dataset:
    print("x = ", x.numpy(), ", y = ", y.numpy())

x = [[4 5 6 7] [1 2 3 4]], y = [[8] [5]]
x = [[3 4 5 6] [2 3 4 5]], y = [[7] [7]]
x = [[5 6 7 8] [0 1 2 3]], y = [[9] [4]]
```

### Feeding windowed datasets into neural networks

```python
def windowed_dataset(series, window_size: int, batch_size: int, shuffle_buffer: int):
    return tf.data.Dataset.from_tensor_slices(series)
        .window(window_size + 1, shift=1, drop_remainder=True)
        .flat_map(lambda window: window.batch(window_size + 1))
        .shuffle(buffer_size=shuffle_buffer)
        .map(lambda window: (window[:-1], window[-1]))
        .batch(batch_size)
        .prefetch(1)
```

### Single-layer neural networks

```python
split_time = 1000
time_train, time_val = time[:split_time], time[split_time:]
x_train, x_val = series[:split_time], series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)

# Linear Regression:
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=.9))
model.fit(dataset, epochs=100, verbose=0)

# Forecasting:
forecast = [model.predict(series[time:time + window_size][np.newaxis]) for time in range(len(series) - window_size)][split_time - window_size:]

results = np.array(forecast)[:, 0, 0]
```

### Deep neural network training, tuning and prediction

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=.9)

model.compile(loss="mse", optimizer=optimizer)

history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

# Plot loss against learning rate, per epoch
lrs = 1e-8 * 10**(np.arange(100) / 20)
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```

If you want to inspect the learned parameters in a layer after training, you can assign a variable to the layer and add it to the model using that variable. Then you can inspect the properties after training.

If you want to amend the learning rate of the optimizer on the fly after each epoch, use a `LearningRateScheduler` **callback**.

## 3. Recurrent Neural Networks for Time Series

RNNs have more than one output: $\hat{Y}, H$.

### Lambda layers

A lambda layer allows allows for execution of arbitrary operations during training, within the model definition itself.

The `windowed_dataset` helper function returns two-dimensional batches of windows on the data, with the first being the batch size and the second the number of timestamps. But an RNN expects three-dimensions: batch size, the number of timestamps, and the series dimensionality. With the Lambda layer, we can fix this without rewriting our `windowed_dataset` helper function. Using the Lambda, we just expand the array by one dimension. By setting input shape to none, we're saying that the model can take sequences of any length.

```python
window_size = 20

model = keras.models.Sequential([
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    keras.layers.SimpleRNN(window_size, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(window_size),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x * 100.0)
])

tf.keras.backend.clear_session()
```

Note the `axis` parameter of `tf.expand_dims` defines the dimension index at which you will expand the shape of the tensor.

`tf.keras.backend.clear_session()` clears out all temporary variables from previous sessions.

Similarly, if we scale up the outputs by 100, we can help training. The default activation function in the RNN layers is $\tanh$, the hyperbolic tangent activation. This outputs values between negative one and one. Since the time series values are in that order usually in the 10s like 40s, 50s, 60s, and 70s, then scaling up the outputs to the same ballpark can help us with learning.

### LSTMs

Perhaps using LSTMs would be a better approach than RNNs.

*In addition to the* $H$ *output, LSTMs also have a cell state that runs across all cells.*

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), input_shape=[None]),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0),
])
```

## 4. Real-World Time Series Data

A combination of neural network architectures (DNNs, Convolutions, RNNs / LSTMs) are often best for predicting time series.

The MAE is a good metric for measuring accuracy of predictions for time series because it doesn't heavily punish larger errors like square errors do.

### Bi-Directional LSTMs

### Convolutions with LSTMs

### Combining analysis tools

Note the input shape for a univariate time series to a `Conv1D` layer is `[None, 1]`.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[None, 1]),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400.0),
])

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=.9)

model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

model.fit(dataset, epochs=500)
```

Note the loss used here is the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).
