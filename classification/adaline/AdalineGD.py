import numpy as np
from typing import Self

class AdalineGD():
    """ADAptive LInear NEuron classifier
    
    eta - Learning rate (between 0.0 and 1.0)
    n_iter - Passes over the training dataset / epoch
    random_state - Random number generator sed for random wieght initialization
    w - weights after fitting    
    cost - sum-of-squares cost function value averaged over all training samples in each epoch

    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w = []
        self.cost = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit training data

        X - Training vectors / matrix of features
        y - Target values / classes
        
        """

        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)

        return self
        

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):
        """Compute linear activation

        In the case of Adaline, it is simple the identity function f(x) = x

        """
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
