import math
from typing import NoReturn

import numpy as np
import pandas
import random
import copy

# Task 1

def cyclic_distance(points, dist):
    answer = 0
    for i in range(len(points)):
        answer += dist(points[i - 1], points[i])
    return answer


def l2_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))


def l1_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist  # Do not change

    def optimize(self, X):
        return self.optimize_explain(X)[-1]

    def optimize_explain(self, X):
        n = X.shape[0]
        current_perm = np.random.permutation(n)

        def swap(i, j):
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]

        def part_cyclic_distance(i, j):
            diff = 0.0
            if i != j + 1:
                diff -= self.dist(X[current_perm[i - 1]], X[current_perm[i]])
                diff += self.dist(X[current_perm[i - 1]], X[current_perm[j]])
                diff -= self.dist(X[current_perm[j]], X[current_perm[(j + 1) % n]])
                diff += self.dist(X[current_perm[i]], X[current_perm[(j + 1) % n]])
            if i != (j - 1 + n) % n:
                diff -= self.dist(X[current_perm[j - 1]], X[current_perm[j]])
                diff += self.dist(X[current_perm[j - 1]], X[current_perm[i]])
                diff -= self.dist(X[current_perm[i]], X[current_perm[(i + 1) % n]])
                diff += self.dist(X[current_perm[j]], X[current_perm[(i + 1) % n]])
            return diff

        explanation = []
        for it in range(self.max_iterations):
            best = None
            best_dist = 0.0
            for i in range(n):
                for j in range(i):
                    current_dist = part_cyclic_distance(i, j)
                    if current_dist < best_dist:
                        best = i, j
                        best_dist = current_dist
            if best is None:
                break
            swap(best[0], best[1])
            explanation.append(current_perm.copy())
        return explanation
        

# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations
        self.ans = None
        self.ans_dist = 1e20

    def optimize(self, X):
        self.optimize_explain(X)
        return self.ans

    def optimize_explain(self, X):
        n = X.shape[0]
        pop = np.zeros((self.pop_size, n), dtype=np.int32)
        for i in range(self.pop_size):
            pop[i] = np.random.permutation(n)
        explanation = []
        for it in range(min(self.iters, 30)):
            extra = []
            for i in range(25):
                crossover = pop[np.random.choice(n, 2)]
                segment = np.sort(np.random.choice(n, 2))
                first = crossover[0][segment[0]:segment[1] + 1]
                second = np.setdiff1d(crossover[1], first, assume_unique=True)
                extra.append(np.concatenate([first, second]))
            pop = np.concatenate([pop, np.array(extra)])
            dists = [cyclic_distance(X[perm], self.dist) for perm in pop]
            d_order = np.argsort(dists)
            pop = pop[d_order[:self.pop_size]]
            if dists[d_order[0]] < self.ans_dist:
                self.ans = pop[0]
                self.ans_dist = dists[d_order[0]]
            explanation.append(pop)
        for _ in range(len(explanation), self.iters):
            explanation.append(explanation[-1])
        return explanation


# Task 4

from collections import Counter


class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        c = Counter()
        words2d = [sentence.split() for sentence in X]
        for words in words2d:
            c.update(words)
        self.counters = c.most_common(voc_limit)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ), 
            который необходимо векторизовать.
        
        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        counters = [Counter(sentence.split()) for sentence in X]
        res = [
            [counter[word] for word, _ in self.counters]
            for counter in counters
        ]
        return np.array(res)

# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Оценивает параметры распределения p(x|y) для каждого y.
        """
        pass

    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу.
            Матрица размера (X.shape[0], n_classes)
        """
        return None
