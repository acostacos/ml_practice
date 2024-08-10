import numpy as np
from typing import Self

class AdalineSGD():
    """ADAptive LInear NEuron classifier
    
    eta - Learning rate (between 0.0 and 1.0)
    n_iter - Passes over the training dataset / epoch
    shuffle - shuffles training data every epoch if True to prevent cycles
    random_state - Random number generator sed for random wieght initialization
    w_initialized - is w initialized?
    w_ - weights after fitting, not initialized in constructor
    cost_ - sum-of-squares cost function value averaged over all training samples in each epoch

    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50,
                 shuffle: bool = True, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit training data

        X - Training vectors / matrix of features
        y - Target values / classes
        
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
        
    def _initialize_weights(self, m: int):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True
    
    def _update_weights(self, xi: np.ndarray, target: float) -> float:
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X) -> np.ndarray:
        """Compute linear activation

        In the case of Adaline, it is simple the identity function f(x) = x

        """
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
