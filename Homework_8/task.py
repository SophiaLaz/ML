from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import numpy as np
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List
from collections import Counter


# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    unique_x, count = np.unique(x, return_counts=True)
    len_x = x.shape[0]
    return np.sum(count * (len_x - count)) / np.power(len_x, 2)


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    unique_x, count = np.unique(x, return_counts=True)
    len_x = x.shape[0]
    return np.log2(len_x) - np.sum(count * np.log2(count)) / len_x


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable, error_all_y: float=None) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    len_left, len_right = len(left_y), len(right_y)
    return (error_all_y or criterion(np.concatenate([left_y, right_y]))) - \
           len_left / (len_left + len_right) * criterion(left_y) - \
           len_right / (len_left + len_right) * criterion(right_y)


# Task 2

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        self.y = self._all_classes(ys)
        dict_labels = Counter(ys.tolist())
        self.dict = {y_i: count_i / ys.shape[0] for y_i, count_i in dict_labels.items()}

    def _all_classes(self, ys: np.ndarray):
        unique_y, count_uniq = np.unique(ys, return_counts=True)
        return unique_y[np.argmax(count_uniq)]

    def predict_proba_leaf(self, X):
        return self.dict


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

    def predict_proba_leaf(self, X: np.ndarray):
        return self.left.predict_proba_leaf(X) if X[self.split_dim] < self.split_value \
            else self.right.predict_proba_leaf(X)


# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """

        def _one_iteration_of_tree(parth_ind, depth):

            if depth == self.max_depth or len(parth_ind) <= self.min_samples_leaf:
                return DecisionTreeLeaf(y[parth_ind])

            best_gain, split_value, split_ind_value, split_dim = -1e12, None, None, None
            error_all_y = 1
            len_ind = len(parth_ind)
            for feature in range(X.shape[1]):
                sort_ind = parth_ind[np.argsort(X[parth_ind, feature])]
                for mid in range(len_ind):

                    if mid <= self.min_samples_leaf or len_ind - mid <= self.min_samples_leaf:
                        continue
                    elif mid == 0 or X[sort_ind[mid]][feature] == X[sort_ind[mid - 1]][feature]:
                        continue

                    elif (IG := gain(y[sort_ind[:mid]], y[sort_ind[mid:]], self.criterion,
                                     error_all_y)) and IG > best_gain:
                        best_gain, split_value, split_ind_value, split_dim = \
                            IG, X[sort_ind[mid]][feature], mid, feature

            if split_value is not None and split_dim is not None:
                sort_ind = parth_ind[np.argsort(X[parth_ind, split_dim])]
                return DecisionTreeNode(
                    split_dim, split_value,
                    _one_iteration_of_tree(sort_ind[:split_ind_value], depth + 1),
                    _one_iteration_of_tree(sort_ind[split_ind_value:], depth + 1)
                )
            else:
                return DecisionTreeLeaf(y[parth_ind])

        self.root = _one_iteration_of_tree(np.arange(X.shape[0]), 1)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """
        return [self.root.predict_proba_leaf(X[i]) for i in range(X.shape[0])]

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


# Task 4
task4_dtc = DecisionTreeClassifier(
                criterion="gini",
                max_depth=6,
                min_samples_leaf=3
)

def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)

if __name__ == '__main__':

    X, y = synthetic_dataset(1000)
    # rfc = RandomForestClassifier(n_estimators=100)
    # rfc.fit(X, y)
    tree = DecisionTreeClassifier(X, y)
    print("Accuracy:", np.mean(tree.predict(X) == y))
    # print("Importance:", feature_importance(rfc))