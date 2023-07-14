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

# [Practical Time Series Analysis](https://www.coursera.org/learn/practical-time-series-analysis)

## R

```bash
sudo apt-get install r-base
```

## Basic Statistics

### Descriptive Statistics

#### Numerical descriptions

```R
data <- c(35, 8, 10, 23, 42)  # concatenation operator
summary(data)
mean(data)
sd(data)
```

#### Graphical descriptions

Notebooks attached.

### Inferential Statistics

#### OLS / Linear Regression

The response variable $Y_i$ depends on the explanatory variable $x_i$ in a linear way:

$$
Y_i = \beta_0 + \beta_1 x_i + \epsilon_i
$$

Assumptions:

- The errors / **residuals** are normally distributed and, on average, zero. Assess normality with QQ Plots:

```R
qqnorm(c02.residuals)
```

- The errors all have the same variance (they are **homoskedastic**).
- The errors are unrelated to each other (they are independent across observations).

OLS: Minimise least squares error $\sum{(\text{Observed} - \text{Predicted})^2}$.

```R
x <- c(1, 2, 3, 4)
y <- c(5, 7, 12, 13)
( m <- lm(y ~ x) )
```

The sum of the residuals in a simple linear regression model is 0.

The sum of the fitted values is equal to the sum of the observed values. Which is why the sum of the residuals is 0.

#### T-tests

```R
t.test(data1, data2, paired=TRUE, alternative="two.sided")
```

Null hypothesis $H_0$: Mean response is the same for both, i.e. $\mu_1 - \mu_2 = 0$

$$
\alpha \equiv P(\text{Type I Error}) = 0.05
$$

$$
t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}
$$

Confidence Error: Estimate $\plusmn$ Table Value $\times$ (Estimated) Standard Error

$$
\bar{d} \plusmn t_{\frac{\alpha}{2}} \times \frac{s}{\sqrt{n}}
$$

The **standard error** is the standard deviation of the standard deviation of a sampling distribution.

#### Covariance

$$
Cov[X, Y] = E[(X - \mu_X)(Y - \mu_Y)]
$$

$$
cov = \frac{1}{n-1} \sum{(x_i - \bar{x})(y_i - \bar{y})}
$$

#### Correlation

$$
\rho(X, Y) = E[(\frac{X - \mu_X}{\sigma_X})(\frac{Y - \mu_Y}{\sigma_Y})]
$$

$$
r = \hat{\rho} = \frac{1}{n-1} \sum{(\frac{x_i - \bar{x}}{s_X})(\frac{y_i - \bar{y}}{s_Y})}
$$

$$
SSX = \sum{(x_i - \bar{x})^2} = \sum{x_i^2} - \frac{1}{n} \sum{x_i} \sum{x_i}
$$

$$
SSY = \sum{(y_i - \bar{y})^2} = \sum{y_i^2} - \frac{1}{n} \sum{y_i} \sum{y_i}
$$

$$
SSXY = \sum{(x_i - \bar{x})(y_i - \bar{y})} = \sum{x_i y_i} - \frac{1}{n} \sum{x_i} \sum{y_i}
$$

$$
\frac{1}{n-1} \sum{(\frac{x_i - \bar{x}}{s_X})(\frac{y_i - \bar{y}}{s_Y})} = \frac{1}{n-1} \sum{(\frac{x_i - \bar{x}}{\sqrt{\frac{SSX}{n-1}}}) (\frac{y_i - \bar{y}}{\sqrt{\frac{SSY}{n-1}}})}
$$

$$
\begin{split}
r & = \bar{\rho} = \sum{(\frac{x_i - \bar{x}}{\sqrt{SSX}}) (\frac{y_i - \bar{y}}{\sqrt{SSY}})} \\
 & = \frac{1}{\sqrt{SSX SSY}} \sum{(x_i - \bar{x}) (y_i - \bar{y})} \\
 & = \frac{SSXY}{\sqrt{SSX} \sqrt{SSY}}
\end{split}
$$

## Visualisation and Modelling of Time Series

### (Weak) Stationarity

- No systematic change in the mean, i.e. no trend
- No systematic change in variance
- Only *periodic* variations

For non-stationary time series, we will do some transformations to get stationary time series.

### Autocovariance Function

A random variable maps from a sample space S to the real numbers, $X: S \rarr \R$.

Covariance is the linear dependence between two random variables, $Cov(X, Y) = E[(X - \mu_X) (Y - \mu_Y)] = Cov(Y, X)$.

A collection of random variables $X_1, X_2, X_3, ...$ is a stochastic process where $X_t \sim dist(\mu_t, \sigma_t^2)$: Unlike a deterministic process, there is some randomness at every time step, and you can never be sure exactly where you are.

A time series is a realisation of a stochastic process.

$$
\gamma(s, t) = Cov(X_s, X_t) = E[(X_s - \mu_s) (X_t - \mu_t)]
$$

$$
\gamma(t, t) = Cov(X_t, X_t) = E[(X_t - \mu_t)^2] = Var(X_t) = \sigma_t^2
$$

Thus, the autocovariance function

$$
\gamma_k = \gamma(t, t+k) \approx c_k
$$

Regardless of the value of $t$, the time difference $k$ is important for autocovariance because the assumption is that the time series is *stationary*, i.e. there is no trend.

### Autocovariance Coefficients

As above, assume weak stationarity.

Thus, $c_k$ is an estimation of $\gamma_k$.

$$
c_k = \frac{\sum_{t=1}^{N-k}{(x_t - \bar{x})(x_{t+k} - \bar{x})}}{N}
$$

Autocovariance coefficients can be calculated at different lags $\gamma_k = Cov(X_t, X_{t+k})$ with the `acf()` R routine.

### Autocorrelation Function

As above, assume weak stationarity.

The autocorrelation coefficient between $X_t$ and $X_{t+k}$ is defined to be

$$
-1 \leq \rho_k = \frac{\gamma_k}{\gamma_0} \leq 1
$$

Thus, estimation of the autocorrelation coefficient at lag $k$

$$
r_k = \frac{c_k}{c_0}
$$

### Random Walks

$$
X_t = X_{t-1} + Z_t, Z_t \sim Normal(\mu, \sigma^2)
$$

where $Z_t$ is the white noise (residual).

$$
X_0 = 0, X_1 = Z_1, X_2 = Z_1 + Z_2, X_t = \sum_{i=1}^t{Z_i}
$$

$$
E[X_t] = E[\sum_{i=1}^t{Z_i}] = \sum_{i=1}^t{E[Z_i]} = \mu t
$$

$$
Var[X_t] = Var[\sum_{i=1}^t{Z_i}] = \sum_{i=1}^t{Var[Z_i]} = \sigma^2 t
$$

**Removing the trend from a random walk:** Given $X_t = X_{t-1} + Z_t$, $X_t - X_{t-1} = Z_t$. Define the difference as $\nabla X_t = X_t - X_{t-1}$ (`diff()` in R). $\nabla X_t$ is a purely random process that is stationary.

### Moving Averages

A moving averages process of order $q$ can be defined as $MA(q)$:

$$
X_t = Z_t + \theta_1 Z_{t-1} + \theta_2 Z_{t-2} + ... + \theta_q Z_{t-q}
$$

$$
Z_i \text{ i.i.d.}, Z_i \sim Normal(\mu, \sigma^2)
$$

The current value of the process is a linear combination of the noises from current and past time steps.

Autocorrelation function of the process, i.e. ACF of MA(q), cuts off and becomes zero at the order of the process, i.e. at lag q.

## Stationarity

A stochastic process is a family of random variables structured with a time index, denoted $X_t$ for discrete processes (e.g. daily high temperatures) and $X(t)$ for continuous processes (e.g. Brownian motion of a particle).

| $X_1$ | $X_2$ | $X_3$ | ... | $X_N$ |
| --- | --- | --- | --- | --- |
| $E[X_1] = \mu_1$ | $E[X_2] = \mu_2$ | $E[X_3] = \mu_3$ |  | $E[X_N] = \mu_N$ |
| $Var[X_1] = \sigma_1^2$ | $Var[X_2] = \sigma_2^2$ | $Var[X_3] = \sigma_3^2$ |  | $Var[X_N] = \sigma_N^2$ |

Therefore, can define a

- mean function: $\mu(t) \equiv \mu_t \equiv E[X(t)]$
- variance function: $\sigma^2(t) \equiv \sigma_t^2 \equiv Var[X(t)]$

Estimation: How can infer the properties of a stochastic process from a single realisation?

**Strict Stationarity:** A process is strictly stationary if the joint distribution of $X(t_1), X(t_2), ..., X(t_k)$ is the same as the joint distribution of $X(t_1 + \tau), X(t_2 + \tau), ..., X(t_k + \tau)$.

Then, the joint distribution depends only on the lag spacing. Therefore, the autocovariance function is defined as:

$$
\gamma(t_1, t_2) = \gamma(t_2 - t_1) = \gamma(\tau)
$$

**Weak Stationarity:** A process is weakly stationary if the mean function $\mu(t) = \mu$ and the ACF $\gamma(t_1, t_2) = \gamma(t_2 - t_1) = \gamma(\tau)$. Therefore, under weak stationarity, *mean and variance are constant.*

This is a much weaker assumption but still useful!

### White Noise

White Noise is stationary!

Consider a discrete family of i.i.d. Normal r.v.s (often Gaussian)

$$
X_t \sim iid N(0, \sigma^2)
$$

The mean function $\mu(t) = 0$ is obviously constant, so consider

$$
\gamma(t_1, t_2) =
\begin{cases}
0 & t_1 \neq t_2 \\
\sigma^2 & t_1 = t_2 \\
\end{cases}
$$

### Random walks

Random walks are not stationary!

$$
X_t = X_{t-1} + Z_t = \sum_{i=1}^t{Z_i}
$$

The mean and variance increase linearly with time:

$$
E[X_t] = E[\sum_{i=1}^t{Z_i}] = \sum_{i=1}^t{E[Z_i]} = t \times \mu
$$

$$
Var[X_t] = Var[\sum_{i=1}^t{Z_i}] = \sum_{i=1}^t{Var[Z_i]} = t \times \sigma^2
$$

### Moving Average Processes

Moving average processes are (weakly) stationary! The mean is constant (equal to zero) and the autocovariance depends just upon lag spacing.

Start with i.i.d. r.v.s $Z_t \sim iid(0, sigma^2)$.

$MA(q)$ process: $X_t = \beta_0 Z_t + \beta_1 Z_{t-1} + ... + \beta_q Z_{t-q}$

$q$ tells us how far back to look along the white noise sequence for our weighted average.

## Backward Shift Operator

If the limit of a sequence exists, i.e. $\lim_{n \rarr \infty} a_n = a$, it is said to be convergent.

If the partial sums $s_n$ of a sequence, i.e. up to the $n^{th}$ element, is convergent to a number $s$, then the infinite series $\sum_{k=1}^{\infty}{a_k}$ is convergent, and is equal to $s$.

$$
\sum_{k=1}^{\infty}{a_k} = \lim_{n \rarr \infty}{s_n} = \lim_{n \rarr \infty}{(a_1 + a_2 + \dots + a_n)} = s
$$

A series is absolutely convergent if $\sum_{k=1}^{\infty}{ \lvert a_k \rvert }$ is convergent. Absolute convergence implies convergence.

Consider a geometric sequence $\{ar^{n-1}\}*{n=1}^{\infty} = \{a, ar, ar^2, ar^3, \dots \}$. A geometric series $\sum*{k=1}^{\infty}{ar^{k-1}}$ converges to $\frac{a}{1-r}$ if $\lvert r \rvert \lt 1$.

Define the backward shift operator $B$ such that

$$
BX_t = X_{t-1}, B^2X_t = BBX_t = BX_{t-1} = X_{t-2}, B^kX_t = X_{t-k}
$$

Thus, a random walk can be expressed as:

$$
X_t = X_{t-1} + Z_t \rArr X_t = BX_t + Z_t \rArr (1-B)X_t = Z_t \rArr \phi(B)X_t = Z_t
$$

where $\phi(B) = 1 - B$, in this example.

### MA(q)

Then, an $MA(q)$ process with a drift $\mu$:

$$
\begin{split}
X_t & = \mu + \beta_0 Z_t + \beta_1 Z_{t-1} + \dots + \beta_q Z_{t-q} \\
 & = \mu + \beta_0 Z_t + \beta_1 B^1 Z_t + \dots + \beta_q B_q Z_t
\end{split}
$$

$$
X_t - \mu = \beta(B) Z_t
$$

$$
\beta(B) = \phi_0 + \phi_1 B + \dots + \phi_q B^q
$$

### AR(p)

$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + Z_t
$$

$$
\begin{split}
Z_t & = X_t - \phi_1 X_{t-1} - \phi_2 X_{t-2} - \dots - \phi_p X_{t-p} \\
 & = X_t - \phi_1 BX_t - \phi_2 B^2X_t - \dots - \phi_p B^pX_t \\
 & = \phi(B) X_t
\end{split}
$$

where $\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p$

### Invertibility

Consider a stochastic process $\{X_t\}$ with random disturbances or white noise $\{Z_t\}$. $\{X_t\}$ is called invertible if $Z_t = \sum_{k=0}^{\infty}{\pi_k X_{t-k}}$ where $\sum_{k=0}^{\infty}{\lvert \pi_k \rvert}$ is convergent.

Invertibility guarantees a unique MA process corresponding to the observed ACF.

### Duality

### Mean Square Convergence

## Autoregressive Processes

A Moving Average process MA(q) starts with white noise $Z_t \sim iid(0, \sigma^2)$ and takes an average of the last $q$ terms:

$$
X_t = \theta_0 Z_t + \theta_1 Z_{t-1} + \dots + \theta_q Z_{t-q}
$$

An Autoregressive Process AR(p), on the other hand, depends on the previous terms in the process:

$$
X_t = Z_t + \phi_1 X_{t-1} + \dots + \phi_p X_{t-p}
$$

Changing $\phi$ has a profound effect on the drop off in the ACF.

A Random Walk is an example of an Autoregressive process.

Note that an autoregressive process may not necessarily be stationary! For example, an AR(1) process is only stationary if $-1 < \phi < 1$.

*An AR(p) process can be expressed as an infinite order MA(q) process.*

## Yule-Walker Equations

### Difference Equations

## Partial Autocorrelation and PACF

A PACF plot tells you the likely order of an AR(p) process.

## Yule-Walker Matrix Notation and AR(p) Model Parameter Estimation

## Akaike Information Criterion for Model Quality

- Give *credit* for models which reduce the error sum of squares
- Build in a *penalty* for models which bring in too many parameters

$$
AIC = -2 \times \log(\text{maximum likelihood}) + 2 \times (\text{number of parameters in the model})
$$

Simple AIC of a given model with $p$ terms:

$$
AIC = \log(\hat{\sigma}^2) + \frac{n + 2 \times p}{n}, \text{ where } \hat{\sigma}^2 = \frac{SSE}{n}
$$

## ARMA

Bring together an MA(q) and an AR(p):

$$
X_t = \text{Noise} + \text{Autoregressive Part} + \text{Moving Average Part}
$$

$$
X_t = Z_t + \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \theta_1 Z_{t-1} + dots + \theta_q Z_{t-q}
$$

ARMA (mixed process):

$$
\theta(B) Z_t = \phi(B) X_t
$$

$$
Z_t = \frac{\phi(B)}{\theta(B)} X_t
$$

## ARIMA

## Forecasting with Smoothing Techniques

### Seasonality

### Single Smoothing

### Double Smoothing

### Triple Exponential Smoothing
