import numpy as np


class OurLogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    @staticmethod
    def sigmoid(x) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Make predictions using the current weights and bias
            # note; @ is a matrix multiplication operator in numpy
            predictions: np.ndarray = OurLogisticRegression.sigmoid(
                X @ self.weights + self.bias
            )

            # Calculate the error
            error: np.ndarray = y - predictions

            # Update the weights and bias
            self.weights += self.learning_rate * error @ X
            self.bias += self.learning_rate * error.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Make predictions using the current weights and bias
        predictions: np.ndarray = OurLogisticRegression.sigmoid(
            X @ self.weights + self.bias
        )
        return predictions > 0.5
