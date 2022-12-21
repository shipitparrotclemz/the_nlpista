from typing import Union

import numpy as np
import scipy
from scipy.sparse import hstack, csr_matrix


class OurLogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.bias_and_weight = None

    @staticmethod
    def sigmoid(x) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def fit(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: Union[np.ndarray, csr_matrix],
    ) -> None:
        """
        Trains the model, by tweaking self.weights and self.bias, such that the model predicts the right labels for the training data
        """
        n_samples, n_features = X.shape

        """
        A bias term is a constant value that is added to the linear combination of the features 
        before applying the sigmoid function. 
        
        It allows the model to shift the decision boundary away from the origin 
        and can improve the model's ability to fit the data.
        
        This one here is the co-efficient, to be multiplied with the actual bias in the weights
        """

        # concatenate a column of one to the input data, as the bias
        # X original shape is (45000,101895), the new shape is (45000, 101896)
        bias_column: csr_matrix = csr_matrix(np.ones((n_samples, 1)))

        print(f"X shape: {X.shape}")
        print(f"bias_column shape: {bias_column.shape}")

        bias_with_X = hstack([bias_column, X])

        """
        In general, it is recommended to initialize the weights in logistic regression with small random values 

        rather than setting them all to zero. 
        
        This can help break symmetry, speed up convergence, and improve generalization performance.
        
        Initialize the weight and bias with random values from a standard normal distribution
            
        The first column is the bias, the rest are the weights
        """

        # the shape should be (101896, 1), the original weight vector shape is (101895, 1) without the bias
        # we use a normal distribution around mean 0, and standard deviation 0.1

        """
        Q: Why mean of 0?
        
        A mean of 0 this can help to ensure that the weights are symmetrically distributed around the origin and
        do not introduce any bias into the model.
        
        Q: Why standard deviation of 0.1?
        
        a smaller standard devation of 0.1 instead of 1.0 can help to prevent the weights from becoming too large or too small
        which can cause problems during training.
        """
        self.bias_and_weight: np.ndarray = np.random.normal(
            0, 0.1, size=(n_features + 1,)
        )
        print(f"bias_and_weight shape: {self.bias_and_weight.shape}")

        for i in range(self.max_iter):
            # Make predictions using the current weights and bias
            # note; @ is a matrix multiplication operator in numpy

            p: np.ndarray = OurLogisticRegression.sigmoid(
                bias_with_X @ self.bias_and_weight
            )

            # Calculate the cost function: cross entropy loss
            # this cost function represents the current model's performance on the training data as it trains
            loss: np.ndarray = (-1 / n_samples) * (
                y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)
            )

            # For every 500 iterations, print the loss
            if i % 500 == 0:
                print(f"Loss at training iteration {i}: {np.mean(loss)}")

            # calculate the gradients of the weights; it should be the derivative of the cost function w.r.t the weights
            dw: np.ndarray = (1 / n_samples) * bias_with_X.T @ (p - y)

            # Update the weights and bias
            self.bias_and_weight -= self.learning_rate * dw

    def predict(self, X: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        n_samples, n_features = X.shape
        # Make predictions using the current weights and bias
        bias_column: csr_matrix = csr_matrix(np.ones((n_samples, 1)))
        bias_with_X = hstack([bias_column, X])
        predictions: np.ndarray = OurLogisticRegression.sigmoid(
            bias_with_X @ self.bias_and_weight
        )
        return predictions > 0.5

    def score(
        self,
        X: Union[np.ndarray, csr_matrix],
        y: Union[np.ndarray, csr_matrix],
    ) -> float:
        """
        Calculates the accuracy of the model
        :param X: the input test data
        :param y: the input expected labels for the input
        :return: returns the accuracy of the model. (TP + TN) / (TP + TN + FP + FN)
        """
        predictions: np.ndarray = self.predict(X)
        return np.sum(predictions == y) / len(y)


"""
Loss at training iteration 0: 1.2653128806518614
Loss at training iteration 500: 0.6034765766677779
Loss at training iteration 1000: 0.5240688246173634
Loss at training iteration 1500: 0.48138480723222593
Loss at training iteration 2000: 0.4530203479831104
Loss at training iteration 2500: 0.4320356620355317
Loss at training iteration 3000: 0.4154920600935688
Loss at training iteration 3500: 0.40189564593990335
Loss at training iteration 4000: 0.3903913238403545
Loss at training iteration 4500: 0.38044633662660643
Loss at training iteration 5000: 0.3717073613720073
Loss at training iteration 5500: 0.36392866489787945
Loss at training iteration 6000: 0.3569326528416909
Loss at training iteration 6500: 0.3505867683593902
Loss at training iteration 7000: 0.3447893207488684
Loss at training iteration 7500: 0.3394604706204915
Loss at training iteration 8000: 0.3345363137187617
Loss at training iteration 8500: 0.3299648783075952
Loss at training iteration 9000: 0.3257033308523096
Loss at training iteration 9500: 0.3217159654968226
Loss at training iteration 10000: 0.3179727196332198
Loss at training iteration 10500: 0.31444805533930803
Loss at training iteration 11000: 0.3111201030271505
Loss at training iteration 11500: 0.3079699972915995
Loss at training iteration 12000: 0.3049813559711737
Loss at training iteration 12500: 0.3021398672805573
Loss at training iteration 13000: 0.29943295937798065
Loss at training iteration 13500: 0.2968495334347601
Loss at training iteration 14000: 0.29437974608640627
Loss at training iteration 14500: 0.2920148306404362
Loss at training iteration 15000: 0.2897469489786843
Loss at training iteration 15500: 0.2875690679857241
Loss at training iteration 16000: 0.2854748557441514
/Users/gohchangmingclement/Desktop/ship_it_parrot/natural_language_processing/imdb_movie_review/logistic_regression_implementation.py:89: RuntimeWarning: divide by zero encountered in log
  y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)
/Users/gohchangmingclement/Desktop/ship_it_parrot/natural_language_processing/imdb_movie_review/logistic_regression_implementation.py:89: RuntimeWarning: invalid value encountered in matmul
  y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)
Loss at training iteration 16500: nan
Loss at training iteration 17000: nan
Loss at training iteration 17500: nan
Loss at training iteration 18000: nan
Loss at training iteration 18500: nan
Loss at training iteration 19000: nan
Loss at training iteration 19500: nan

Q: Why does the training intermittently blow up?
"""
