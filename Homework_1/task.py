import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas as pd
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    data = pd.read_csv(path_to_csv, delimiter=',')  # разделитель
    all_values = data.values

    np.random.shuffle(all_values)

    x = all_values[:, 1:]  # матрица признаков
    y = all_values[:, 0]  # бинарные метки
    for i in range(len(y)):
        if y[i] == "M":
            y[i] = 1
        else:
            y[i] = 0

    return (x, y)


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    data = pd.read_csv(path_to_csv, delimiter=',')
    all_values = data.values

    np.random.shuffle(all_values)

    x = all_values[:, :-1]
    y = all_values[:, -1]

    return (x, y)


# Task 2


def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    i = int(ratio * len(X))
    X_train, X_test = X[:i], X[i:]
    y_train, y_test = y[:i], y[i:]

    return (X_train, y_train, X_test, y_test)


# Task 3


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    arr, count = {}, 0
    precision, recall = np.array([]), np.array([])

    for s in enumerate(y_true):
        arr[s[1]] = (0, 0, 0, 0)  # (True Positive, False Positive, False Negative, True Negative)

    for i, _ in enumerate(y_true):
        pred = y_pred[i]
        true = y_true[i]
        # подсчёт количества совпадений предсказаний с истинными значениями
        if pred == true:
            count += 1

        for res in arr:
            if res == true:
                if pred == true:
                    arr[res] = (arr[res][0] + 1, arr[res][1], arr[res][2], arr[res][3])  # True Positive
                else:
                    arr[res] = (arr[res][0], arr[res][1], arr[res][2] + 1, arr[res][3])  # False Negative
            elif pred == true:
                arr[res] = (arr[res][0], arr[res][1], arr[res][2], arr[res][3] + 1)  # True Negative
            else:
                if res == pred:
                    arr[res] = (arr[res][0], arr[res][1] + 1, arr[res][2], arr[res][3])  # False Positive
                else:
                    arr[res] = (arr[res][0], arr[res][1], arr[res][2], arr[res][3] + 1)  # True Negative

    for res in sorted(arr):
        p = arr[res][0] / (arr[res][0] + arr[res][1])
        precision = np.append(precision, p)
        r = arr[res][0] / (arr[res][0] + arr[res][2])
        recall = np.append(recall, r)

    accuracy = count / len(y_true)

    return (precision, recall, accuracy)


# Task 4
class Tree_Leaves:
    def __init__(self, X, indexes):
        self.X = X
        self.indexes = indexes

    def query(self, x, k):
        distance = []
        for i, row in enumerate(self.X):
            dist = np.linalg.norm(x - row)
            distance.append(dist)

        array = np.array([distance, self.indexes])
        sortedArray = array[:, array[0].argsort()]

        return sortedArray[:, :k]


def combine(left_result, right_result, k):
    array = np.concatenate((left_result, right_result), axis=1)
    transponed = array.T
    id = np.lexsort((transponed[:, 1], transponed[:, 0]))
    transponed = transponed[id]
    sort_array = transponed.T

    return sort_array[:, :k]


class Tree_Node:
    def __init__(self, X: np.array, indexes, leaf_size):
        self.split_dim = -1
        self.X = X
        self.indexes = indexes
        self.median = -1

        i, greater, lower, cut, length = 0, 0, 0, False, len(X[0])
        merge_sort = np.insert(X, length, indexes, axis=1)

        while i < length:
            idx = np.random.randint(0, length)
            column = X[:, idx]
            median = np.median(column)

            greater = merge_sort[(merge_sort[:, idx] >= median)]
            lower = merge_sort[(merge_sort[:, idx] < median)]

            if len(greater) >= leaf_size and len(lower) >= leaf_size:
                cut = True
                self.split_dim = idx
                self.median = median
                break
            else:
                i += 1

        if cut: # cut - разделение
            lower_id = lower[:, -1]
            self.left = Tree_Node(lower[:, 0:-1], lower_id, leaf_size)
            greater_id = greater[:, -1]
            self.right = Tree_Node(greater[:, 0:-1], greater_id, leaf_size)
        else:
            self.leaf = Tree_Leaves(X, indexes)

    def query(self, x, k):
        if hasattr(self, 'leaf'):
            return self.leaf.query(x, k)
        else:
            dimension = self.split_dim
            median = self.median

            left_array, right_array = [], []

            if x[dimension] > median:
                right_array = self.right.query(x, k)
                right_dist = right_array[0]
                m = max(right_dist)
                to_median = x[dimension] - median
                if to_median < m or len(right_array) < k:
                    left_array = self.left.query(x, k)

            else:
                left_array = self.left.query(x, k)
                left_dist = left_array[0]
                m = max(left_dist)
                to_median = median - x[dimension]
                if to_median < m or len(left_array) < k:
                    right_array = self.right.query(x, k)

            if len(left_array) != 0 and len(right_array) != 0:
                result = combine(left_array, right_array, k)
            elif len(left_array) == 0:
                result = right_array
            elif len(right_array) == 0:
                result = left_array

            return result


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.rootTreeNode = Tree_Node(X, np.arange(0, len(X)), leaf_size)

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k): 
            индексы k ближайших соседей для всех точек из X.

        """
        answer = []
        for search in X:
            k_closests = self.rootTreeNode.query(search, k)
            answer.append(k_closests[1].tolist())

        return answer


# Task 5


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.KD_Tree = KDTree(X, self.leaf_size)
        self.X = X
        self.y = y

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
        all_classes, nearest_neighbors, answer = np.unique(self.y), self.KD_Tree.query(X, 10), []
        for idx in enumerate(X):
            nearest_neighbors_idxes = list(map(int, nearest_neighbors[idx[0]]))
            classes_of_point = np.unique([self.y[i] for i in nearest_neighbors_idxes], return_counts=True)
            probability = np.array([])
            for clazz in all_classes:
                id = np.where(classes_of_point[0] == clazz)
                if np.size(id) > 0:
                    probability = np.append(probability, classes_of_point[1][id] / 10)
                else:
                    probability = np.append(probability, 0)
            answer.append(probability)
        return answer

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        answer = np.argmax(self.predict_proba(X), axis=1)
        return answer
