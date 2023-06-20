import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False


# Task 1


class LinearSVM:
    def __init__(self, C: float):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.

        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None
        self.n = None
        self.x, self.y = None, None
        self.alpha = None

    def _all_alpha(self, X: np.ndarray, y: np.ndarray):
        self.n = y.shape[0]
        P = matrix(np.array([y * y[i] * self._kernel(X, X[i]) for i in range(self.n)]))
        q = matrix(-1.0, (self.n, 1))
        I = np.diag(np.ones(self.n))
        G = matrix(np.concatenate((-I, I), axis=0))
        h = matrix(np.transpose([[0.0] * self.n + [self.C] * self.n]), (2 * self.n, 1))
        A = matrix(y.astype('float'), (1, self.n))
        b = matrix([0.0])
        return np.transpose(np.array(solvers.qp(P, q, G, h, A, b)['x'])).reshape(self.n, )

    def _kernel(self, X: np.ndarray, X_i: np.ndarray):
        return X.dot(X_i)

    def _w(self, X: np.ndarray):
        ans = np.zeros((X.shape[0],), dtype=X.dtype)
        for j in range(self.x.shape[0]):
            ans += self.alpha[j] * self.y[j] * self._kernel(X, self.x[j])
        return ans

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        alpha = self._all_alpha(X, y)
        support_index = 1e-6 < alpha
        self.alpha = alpha[support_index]
        self.x = X[support_index]
        self.y = y[support_index]
        self.support = [i for i in range(len(support_index)) if support_index[i] and alpha[i] < self.C - 1e-6]
        # self.support = [index for index, a in enumerate(self.alpha) if a < self.C - 1e-6]
        self.w = np.sum(self.x * (self.alpha * self.y).reshape(len(self.y), 1), axis=0)
        self.b = self._w(X[self.support[0]][None, :]) - y[self.support[0]]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        return np.matmul(self.w, X.T) - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))
    
# Task 2

def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"
    return lambda X, y: np.power(c + np.dot(X, y), power)

def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"
    return lambda X, y: np.exp(-sigma * np.power(np.linalg.norm(X - y, axis=1), 2))


# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.

        """
        self.kernel = kernel
        self.C = C
        self.w = None
        self.b = None
        self.support = None
        self.n = None
        self.x, self.y = None, None
        self.alpha = None

    def _all_alpha(self, X: np.ndarray, y: np.ndarray):
        self.n = y.shape[0]
        P = matrix(np.array([y * y[i] * self.kernel(X, X[i]) for i in range(self.n)]))
        q = matrix(-1.0, (self.n, 1))
        I = np.diag(np.ones(self.n))
        G = matrix(np.concatenate((-I, I), axis=0))
        h = matrix(np.transpose([[0.0] * self.n + [self.C] * self.n]), (2 * self.n, 1))
        A = matrix(y.astype('float'), (1, self.n))
        b = matrix([0.0])
        return np.transpose(np.array(solvers.qp(P, q, G, h, A, b)['x'])).reshape(self.n, )

    def _w(self, X: np.ndarray):
        ans = np.zeros((X.shape[0],), dtype=X.dtype)
        for j in range(self.x.shape[0]):
            ans += self.alpha[j] * self.y[j] * self.kernel(X, self.x[j])
        return ans

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        alpha = self._all_alpha(X, y)
        support_index = 1e-6 < alpha
        self.alpha = alpha[support_index]
        self.x = X[support_index]
        self.y = y[support_index]
        self.support = [i for i in range(len(support_index)) if support_index[i] and alpha[i] < self.C - 1e-6]
        self.w = np.sum(self.x * (self.alpha * self.y).reshape(len(self.y), 1), axis=0)
        self.b = self._w(X[self.support[0]][None, :]) - y[self.support[0]]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        return self._w(X) - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))