from typing import Union

import numpy as np
from scipy.sparse import hstack, csr_matrix

from utils.memory_utils import get_memory_usage


class OurLogisticRegression:
    def __init__(
        self, learning_rate: float = 0.01, max_iter: int = 100, lambda_: float = 1
    ):
        """
        Lambda here is the regularization parameter
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.bias_and_weight = None
        self.lambda_ = lambda_

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

        # n_samples: 45000; we have 45,000 training samples
        print(f"n_samples: {n_samples}")
        # n_features: 101895; bag of words has 101895 features
        print(f"n_features: {n_features}")

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

        # X shape: (45000, 101895)
        print(f"X shape: {X.shape}")
        # bias_column shape: (45000, 1)
        print(f"bias_column shape: {bias_column.shape}")

        bias_with_X = hstack([bias_column, X])

        print(f"bias_with_X shape: {bias_with_X.shape}")

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
        
        a smaller standard deviation of 0.1 instead of 1.0 can help to prevent the weights from becoming too large or too small
        which can cause problems during training.
        """

        self.bias_and_weight: np.ndarray = np.random.normal(
            0, 0.1, size=(n_features + 1,)
        )
        # bias_and_weight shape: (101896,)
        print(f"bias_and_weight shape: {self.bias_and_weight.shape}")

        for i in range(self.max_iter):

            get_memory_usage(f"At start of iteration {i}")

            # Make predictions using the current weights and bias
            # note; @ is a matrix multiplication operator in numpy

            p: np.ndarray = OurLogisticRegression.sigmoid(
                bias_with_X @ self.bias_and_weight
            )

            get_memory_usage(f"After getting probabilities at iteration {i}")

            # p shape: (45000,)
            # commented this as printing this shape during every training iteration is too much
            # print(f"p shape: {p.shape}")

            """
            RuntimeWarning: divide by zero encountered in log y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)
            
            It is possible to encounter this intermittent error when p happens to be 0
            
            This clipping of p (probabilities of positive class from sigmoid) will ensure that the values of p are always within the range [1e-10, 1 - 1e-10], which will prevent the log() function from being called with zero or one arguments.
            """
            p = np.clip(p, 1e-10, 1 - 1e-10)

            get_memory_usage(f"After clipping p at iteration {i}")

            # Calculate the cost function: cross entropy loss
            # this cost function represents the current model's performance on the training data as it trains

            """
            scikit learn's logistic regression adds a l2 reg cost to penalize the model for having high coefficients on each bag of words feature
            this is the regularization term, which is the sum of the squares of the weights multiplied by the regularization parameter lambda            
            """
            l2_reg_cost: np.ndarray = (
                (self.lambda_ / (2 * n_samples))
                * self.bias_and_weight[1:]
                @ self.bias_and_weight[1:]
            )

            get_memory_usage(f"After getting l2 regularization cost at iteration {i}")

            # commented this as printing this shape during every training iteration is too much
            # print(f"l2_reg_cost shape: {l2_reg_cost.shape}")

            cost_function: np.ndarray = (-1 / n_samples) * (
                y.T @ np.log(p) + (1 - y).T @ np.log(1 - p)
            ) + l2_reg_cost

            get_memory_usage(f"After getting cost function at iteration {i}")

            # print(f"cost_function shape: {cost_function.shape}")

            # For every 500 iterations, print the loss
            if i % 500 == 0:
                print(
                    f"cost function at training iteration {i}: {np.mean(cost_function)}"
                )

            # calculate the gradients of the weights; it should be the derivative of the cost function w.r.t the weights
            l2_reg_gradient: np.ndarray = (
                self.lambda_ / n_samples
            ) * self.bias_and_weight[1:]

            get_memory_usage(
                f"After getting l2 regularization term in weights gradient at iteration {i}"
            )

            # the first column is the bias, the rest are the weights
            bias_gradient: np.ndarray = (1 / n_samples) * ((p - y) @ bias_with_X[:, 0])

            get_memory_usage(f"After getting bias gradient at iteration {i}")

            weight_gradient: np.ndarray = (1 / n_samples) * (
                (p - y) @ bias_with_X[:, 1:]
            ) + l2_reg_gradient

            get_memory_usage(f"After getting weights gradient at iteration {i}")

            grad = np.concatenate([bias_gradient, weight_gradient], axis=1)
            # Update the weights and bias
            self.bias_and_weight -= self.learning_rate * grad

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
Without L2 Regularization

Loss at training iteration 0: 1.1477796143218801
Loss at training iteration 500: 0.5794130605690193
Loss at training iteration 1000: 0.5087502304621427
Loss at training iteration 1500: 0.4700949095884881
Loss at training iteration 2000: 0.44413081991184916
Loss at training iteration 2500: 0.4247633130383043
Loss at training iteration 3000: 0.40939149203459513
Loss at training iteration 3500: 0.39669068715020067
Loss at training iteration 4000: 0.38589995343427763
Loss at training iteration 4500: 0.37654283902540564
Loss at training iteration 5000: 0.36830131093310814
Loss at training iteration 5500: 0.3609523753168485
Loss at training iteration 6000: 0.35433348644939794
Loss at training iteration 6500: 0.3483224033724736
Loss at training iteration 7000: 0.34282482062618647
Loss at training iteration 7500: 0.3377664271107531
Loss at training iteration 8000: 0.333087611883416
Loss at training iteration 8500: 0.328739819411324
Loss at training iteration 9000: 0.32468297102190025
Loss at training iteration 9500: 0.32088359874426275
Loss at training iteration 10000: 0.3173134693952122
Loss at training iteration 10500: 0.3139485539961887
Loss at training iteration 11000: 0.31076824412523285
Loss at training iteration 11500: 0.3077547464786488
Loss at training iteration 12000: 0.3048926073252724
Loss at training iteration 12500: 0.30216833313148495
Loss at training iteration 13000: 0.2995700839023426
Loss at training iteration 13500: 0.2970874227238508
Loss at training iteration 14000: 0.2947111095623676
Loss at training iteration 14500: 0.2924329304060173
Loss at training iteration 15000: 0.29024555490927556
Loss at training iteration 15500: 0.28814241718980005
Loss at training iteration 16000: 0.2861176155358214
Loss at training iteration 16500: 0.2841658276335227
Loss at training iteration 17000: 0.28228223858935786
Loss at training iteration 17500: 0.2804624795492174
Loss at training iteration 18000: 0.2787025751364491
Loss at training iteration 18500: 0.2769988982665134
Loss at training iteration 19000: 0.2753481311640164
Loss at training iteration 19500: 0.2737472316207584
"""
