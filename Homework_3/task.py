from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.linalg as la
import random
import matplotlib.pyplot as plt
import matplotlib
import copy


# Task 1

def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    return sum([(y_predicted[i] - y_true[i]) ** 2 for i in range(len(y_true))]) / len(y_true)


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    u = sum([(y_predicted[i] - y_true[i]) ** 2 for i in range(len(y_true))])
    y_mean = sum(y_true) / len(y_true)
    v = sum([(y_mean - y_true[i]) ** 2 for i in range(len(y_true))])
    return 1 - u / v


# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here
        # n - count of samples, m - count of features
        self.n, self.m = 0, 0

    def _all_X(self, X: np.ndarray):
        self.n, self.m = X.shape[0], X.shape[1] + 1
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=float)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._all_X(X)
        X_tr = X.T
        self.weights = np.dot(np.dot(la.inv(np.dot(X_tr, X)), X_tr), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._all_X(X)
        return np.dot(X, self.weights)


# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.weights = None  # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.lasso_par = l
        self.n, self.m = 0, 0

    def _all_X(self, X: np.ndarray):
        self.n, self.m = X.shape[0], X.shape[1]
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=float)))

    def _h(self, X: np.ndarray):
        return np.dot(X, self.weights)

    def _grad(self, X: np.ndarray, y: np.ndarray):
        return 2 * np.dot(X.T, self._h(X) - y)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._all_X(X)
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            self.weights -= (self.alpha * (self._grad(X, y)) + self.lasso_par * np.sign(self.weights)) / self.n

    def predict(self, X: np.ndarray):
        X = self._all_X(X)
        return np.dot(X, self.weights)


# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[1:])


def get_most_important_features(linear_regression):
    return list(reversed(np.argsort(get_feature_importance(linear_regression))))
