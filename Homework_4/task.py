import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.n, self.m = 0, 0
        # n - samples, m - features
        self.w = None
        self.iterations = iterations
        self.tags = None

    def _all_X(self, X: np.ndarray):
        self.n, self.m = X.shape[0], X.shape[1] + 1
        return np.hstack((np.ones((X.shape[0], 1), dtype=float), X))

    def _h(self, X: np.ndarray):
        return np.dot(X, self.w)

    def _new_y(self, y: np.ndarray):
        # y[i] -> -1 or 1 : tags
        self.tags = np.unique(y)
        return np.array([-1 if y[i] == self.tags[0] else 1 for i in range(self.n)])

    def _h(self, X: np.ndarray):
        return np.dot(X, self.w)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        X, y = self._all_X(X), self._new_y(y)
        self.w = np.zeros((X.shape[1], ), )
        for _ in range(self.iterations):
            h = np.sign(self._h(X))
            mask = (y != h)
            self.w += np.sum(X[mask] * y[mask, np.newaxis], axis=0)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        h = self._h(self._all_X(X))
        return np.array([self.tags[0] if h[i] < 0 else self.tags[1] for i in range(self.n)])
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.n, self.m = 0, 0
        # n - samples, m - features
        self.w = None
        self.iterations = iterations
        self.tags = None

    def _all_X(self, X: np.ndarray):
        self.n, self.m = X.shape[0], X.shape[1] + 1
        return np.hstack((np.ones((X.shape[0], 1), dtype=float), X))

    def _h(self, X: np.ndarray):
        return np.dot(X, self.w)

    def _new_y(self, y: np.ndarray):
        # y[i] -> -1 or 1 : tags
        self.tags = np.unique(y)
        return np.array([-1 if y[i] == self.tags[0] else 1 for i in range(self.n)])
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        X, y = self._all_X(X), self._new_y(y)
        self.w = np.zeros((X.shape[1],), )
        best_count, best_w = self.n, np.copy(self.w)
        for _ in range(self.iterations + 1):
            h = np.sign(self._h(X))
            mask = (y != h)
            if np.sum(mask) < best_count:
                best_count, best_w = np.sum(mask), np.copy(self.w)
            self.w += np.sum(X[mask] * y[mask, np.newaxis], axis=0)
        self.w = np.copy(best_w)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        h = self._h(self._all_X(X))
        return np.array([self.tags[0] if h[i] < 0 else self.tags[1] for i in range(self.n)])
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Первый элемент для каждой картинки: сумма дисперсии по столбцам,
        второй элемент - сумма дисперсий по строкам.
        Её размерность: (n_images, 2).
    """
    features = []
    for image in images:
        var_col = np.var(image, axis=0)
        var_rows = np.var(image, axis=1)
        features.append([np.sum(var_col), np.sum(var_rows)])
    return features