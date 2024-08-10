import numpy as np
from typing import Self

class Perceptron():
    """Perceptron classifier.

    eta - Learning rate (between 0.0 and 1.0)
    n_iter - Passes over the training dataset / epoch
    random_state - Random number generator sed for random wieght initialization
    w - weights after fitting    
    errors - number of misclassifications (updates) in each epoch

    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w = []
        self.errors = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit training data

        X - Training vectors / matrix of features
        y - Target values / classes
        
        """

        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)

            self.errors.append(errors)

        return self
        

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
