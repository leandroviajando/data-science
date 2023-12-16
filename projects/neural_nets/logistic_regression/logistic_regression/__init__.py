import numpy as np


class LogisticModel:
    def __init__(self, num_features):
        """
        You need to set num_features (number of features) when initializing.
        """
        self.W = np.reshape(np.random.randn((num_features)), (num_features, 1))
        self.b = np.zeros((1, 1))
        self.num_features = num_features
        self.losses = []
        self.accuracies = []

    def summary(self):
        """
        Displays Hyperparameters and parameters info
        """
        print("=================================")
        print("Number of features:", self.num_features)
        print("Shape of weights:", self.W.shape)
        print("Shape of biases:", self.b.shape)
        print("=================================")

    def _forward_pass(self, X, Y=None):
        batch_size = X.shape[0]
        Z = np.dot(X, self.W) + self.b
        A = 1.0 / (1.0 + np.exp(-Z))
        loss = float(1e5)
        if Y is not None:
            loss = -1 * np.sum(np.dot(np.transpose(Y), np.log(A)) + np.matmul(np.transpose(1 - Y), np.log(1 - A)))
            loss /= batch_size
        return A, loss

    def _backward_pass(self, A, X, Y):
        batch_size = X.shape[0]
        dZ = A - Y
        dW = np.dot(np.transpose(X), dZ) / batch_size
        db = np.sum(dZ) / batch_size
        return dW, db

    def _update_params(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db

    def predict(self, X, Y=None):
        """
        Given an X, returns (predictions, None). Returns (predictions, loss) value as well if given both X, Y.
        """
        A, loss = self._forward_pass(X, Y)
        Y_hat = A > 0.5
        return np.squeeze(Y_hat), loss

    def evaluate(self, X, Y):
        """
        Given X and Y, returns (accuracy, loss).
        """
        Y_hat, loss = self.predict(X, Y)
        accuracy = np.sum(Y_hat == np.squeeze(Y)) / X.shape[0]
        return accuracy, loss

    def train(self, batch_size, get_batch, lr, iterations, X_train, Y_train, X_test, Y_test):
        """
        Requires:
        batch_size - mini batch size
        get_batch (a function that takes X, Y and returns X_batch, Y_batch) - Used to generate training and validation batches
        lr - learning rate
        iterations - epochs
        X_train, Y_train - training dataset
        X_test, Y_test - data used for validation
        """
        print("Training..")
        self.accuracies = []
        self.losses = []

        for i in range(0, iterations):
            X, Y = get_batch(X_train, Y_train, batch_size)
            A, _ = self._forward_pass(X, Y)
            dW, db = self._backward_pass(A, X, Y)
            self._update_params(dW, db, lr)

            X, Y = get_batch(X_test, Y_test, batch_size)
            val_acc, val_loss = self.evaluate(X, Y)
            self.accuracies.append(val_acc)
            self.losses.append(val_loss)

            print("Iter: {}, Val Acc: {:.3f}, Val Loss: {:.3f}".format(i, val_acc, val_loss))

        print("Training finished.")
