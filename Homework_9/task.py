from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier


from collections import Counter
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 0

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


# Task 1


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


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.X, self.y = X, y
        self.root = None
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        from math import sqrt
        self.max_features = int(sqrt(X.shape[1])) if max_features == "auto" else max_features

        rand_indexes = random.choices(np.arange(X.shape[0]), k=X.shape[0])

        def fit(X: np.ndarray, y: np.ndarray) -> NoReturn:

            def _one_iteration_of_tree(parth_ind, depth):
                parth_ind = np.array(parth_ind)

                if depth == self.max_depth or len(parth_ind) <= self.min_samples_leaf:
                    return DecisionTreeLeaf(y[parth_ind])

                if self.max_features < X.shape[1]:
                    rand_features = random.sample(list(np.arange(X.shape[1])), self.max_features)
                else:
                    rand_features = np.arange(X.shape[1])

                best_gain, split_value, split_ind_value, split_dim = 0.0, None, None, None
                error_all_y = self.criterion(y[parth_ind])
                len_ind = parth_ind.shape[0]
                for feature in rand_features:
                    mask = (X[parth_ind, feature] != np.zeros(len_ind))
                    mid = len(parth_ind[mask])
                    if mid < self.min_samples_leaf or len_ind - mid < self.min_samples_leaf:
                        continue
                    else:
                        IG = gain(y[parth_ind[~mask]], y[parth_ind[mask]], self.criterion, error_all_y)
                        if IG > best_gain:
                            best_gain, split_value, split_ind_value, split_dim = \
                                IG, 1, mid, feature

                if split_value is not None and split_dim is not None:
                    mask = (X[parth_ind, split_dim] != np.zeros(parth_ind.shape[0]))
                    return DecisionTreeNode(
                        split_dim, split_value,
                        _one_iteration_of_tree(parth_ind[~mask], depth + 1),
                        _one_iteration_of_tree(parth_ind[mask], depth + 1)
                    )
                else:
                    return DecisionTreeLeaf(y[parth_ind])

            self.root = _one_iteration_of_tree(rand_indexes, 1)

        fit(X, y)

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
        return np.array([max(p.keys(), key=lambda k: p[k]) for p in proba])
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.tree = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            self.tree.append(DecisionTree(X, y, self.criterion, self.max_depth, self.min_samples_leaf, self.max_features))

    def predict(self, X):
        pred_y = [[0 for _ in range(self.n_estimators)] for _ in range(X.shape[0])]
        ans = [0 for _ in range(X.shape[0])]
        for i in range(self.n_estimators):
            pred_i = self.tree[i].predict(X)
            for j in range(X.shape[0]):
                pred_y[j][i] = pred_i[j]
        from collections import Counter
        for j in range(X.shape[0]):
            ans[j] = Counter(pred_y[j]).most_common(1)[0][0]
        return ans
    
# Task 3

def feature_importance(rfc):
    pass

# Task 4

rfc_age = RandomForestClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_leaf=25,
    max_features=5,
    n_estimators=30
)
rfc_gender = RandomForestClassifier(
    criterion="gini",
    max_depth=7,
    min_samples_leaf=20,
    max_features=15,
    n_estimators=45
)

# Task 5
catboost_rfc_age = CatBoostClassifier()
catboost_rfc_gender = CatBoostClassifier()

catboost_rfc_age.load_model(__file__[:-7] + "catboost_age.cbm")
catboost_rfc_gender.load_model(__file__[:-7] + "catboost_gender.cbm")
