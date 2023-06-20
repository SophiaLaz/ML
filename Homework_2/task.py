from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """

        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.

        """

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = []

    @staticmethod
    def distance(x, y):
        sq_dist = 0
        for x_i, y_i in zip(x, y):
            sq_dist += (x_i - y_i) ** 2
        dist = sq_dist ** 0.5
        return dist

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать
            параметры X и y, даже если y не используется).

        """
        if self.init == "random":
            self.centroids = np.random.randn(self.n_clusters, np.shape(X)[1])
        elif self.init == "sample":
            self.centroids = random.sample(X.tolist(), self.n_clusters)
        else:
            n = self.n_clusters - 1
            ind = np.arange(np.shape(X)[0])
            point_id = np.random.choice(ind, size=1, replace=False)
            point = X[point_id]
            select_point = point[0]

            self.centroids.append(select_point)

            for i in range(n):
                distances = []
                for point in X:
                    dist = self.distance(point, select_point) # обращение к ф-ции
                    distances.append(dist)
                max_dist_id = distances.index(max(distances))
                select_point = X[max_dist_id]

                self.centroids.append(select_point)

    def weigth(self, clusters):
        w = []
        for cluster in clusters:
            cluster_len = len(cluster)
            new_coordinates = [sum(x) for x in zip(*cluster)]
            new_core = [x / cluster_len for x in new_coordinates]
            w.append(new_core)
        self.centroids = w

    def choose_cluster(self, X):
        centers = []
        for x in X:
            distances = []
            for c in self.centroids:
                dist = self.distance(x, c)
                distances.append(dist)
            min_dist = min(distances)
            cluster_id = distances.index(min_dist)
            centers.append(cluster_id)
        return centers

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера,
        к которому относится данный элемент.

        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.

        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров
            (по одному индексу для каждого элемента из X).

        """
        cluster_id = None
        new_cluster = self.choose_cluster(X)

        counter = 0

        while counter <= self.max_iter:
            if new_cluster != cluster_id or cluster_id is None:
                cluster_id, X1 = new_cluster, X.tolist()
                x_in_clusters = [random.sample(X1, self.n_clusters) for _ in range(self.n_clusters)]
                for center in cluster_id:
                    x_in_clusters[center].append(X[cluster_id.index(center)])

                self.weigth(x_in_clusters)
                new_cluster = self.choose_cluster(X)
                counter += 1

            elif new_cluster == cluster_id:
                return cluster_id

        return new_cluster


# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

        self.nearest = []
        self.sample_center = []
        self.clusters = []

    def dfs(self, center_id):
        for x in self.nearest[center_id]:
            x_cluster = self.clusters[x]
            self.clusters[x] = self.clusters[center_id]
            if x_cluster != self.clusters[x] and self.sample_center[center_id]:
                self.dfs(x)

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        KD_Tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        self.nearest = KD_Tree.query_radius(X, self.eps)  # list of np.ndarray of neigh indices
        count_of_neigh, sample_center = [vertex.size for vertex in self.nearest], []

        self.sample_center = []
        for vertex in count_of_neigh:
            if vertex >= self.min_samples:
                self.sample_center.append(True)
            else:
                self.sample_center.append(False)

        for neighbours in count_of_neigh:
            if neighbours >= self.min_samples:
                sample_center.append(count_of_neigh.index(neighbours))
        count_of_clusters, self.clusters = 0, np.zeros(np.shape(X)[0])
        for center in sample_center:
            if self.clusters[center] == 0:
                count_of_clusters += 1
                self.clusters[center] = count_of_clusters
                self.dfs(center)
        return self.clusters

# Task 3


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distances = None
        self.clusters = []

    def dist_between_clusters(self):
        if self.linkage == 'single':
            merge = np.argmin(self.distances)
            point_1 = merge // len(self.clusters)
            point_2 = merge % len(self.clusters)
        elif self.linkage == 'complete':
            merge = np.unravel_index(np.argmax(self.distances), self.distances.shape)


        right, left = min(point_1, point_2), max(point_1, point_2)

        pair = zip(self.distances[left], self.distances[right])
        self.distances[left] = [max(x) for x in pair]
        self.distances[left][left] = 9999999999
        self.distances[:, left] = self.distances[left, :]

        self.distances = np.delete(self.distances, right, axis=0)
        self.distances = np.delete(self.distances, right, axis=1)

        self.clusters[left].extend(self.clusters[right])
        self.clusters.remove(self.clusters[right])
    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        x, labels = np.arange(np.shape(X)[0]), []
        for x_i in x:
            self.clusters.append([x_i])

        self.distances = np.array(sp.spatial.distance.cdist(X, X, metric='euclidean'))
        diagonal = np.eye(np.shape(X)[0]) * 99999999999
        self.distances = self.distances + diagonal
        while len(self.clusters) > self.n_clusters:
            self.dist_between_clusters()

        for x_i in x:
            for cluster in self.clusters:
                if x_i in cluster:
                    labels.append(self.clusters.index(cluster))

        return np.array(labels)
