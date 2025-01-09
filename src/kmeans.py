import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class KMeansState:
    """记录每次迭代的状态"""
    centroids: np.ndarray
    labels: np.ndarray
    iteration: int
    inertia: float


class KMeans:
    def __init__(self, k: int, max_iter: int = 100, init: str = 'random'):
        self.k = k
        self.max_iter = max_iter
        self.init = init

    def fit(self, X: np.ndarray) -> List[KMeansState]:
        """执行聚类并返回每次迭代的状态"""
        # 初始化质心
        centroids = self._initialize_centroids(X)
        states = []

        for i in range(self.max_iter):
            # 分配点到最近的质心
            labels = self._assign_clusters(X, centroids)

            # 更新质心位置
            new_centroids = self._update_centroids(X, labels)

            # 计算惯性（总平方距离）
            inertia = self._compute_inertia(X, labels, new_centroids)

            # 记录当前状态
            states.append(KMeansState(
                centroids=new_centroids.copy(),
                labels=labels,
                iteration=i,
                inertia=inertia
            ))

            # 检查收敛
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return states

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.init == 'random':
            idx = np.random.choice(len(X), self.k, replace=False)
            return X[idx].copy()
        elif self.init == 'k-means++':
            return self._kmeans_plus_plus_init(X)
        raise ValueError(f"Unknown initialization method: {self.init}")

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
        return centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray,
                         centroids: np.ndarray) -> float:
        return np.sum((X - centroids[labels]) ** 2)

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ 初始化策略"""
        centroids = np.empty((self.k, X.shape[1]))
        # 1) 随机选择第一个质心
        first_index = np.random.choice(len(X))
        centroids[0] = X[first_index]

        # 2) 后面的质心基于概率分布选取
        for i in range(1, self.k):
            distances = np.min([np.linalg.norm(X - c, axis=1)
                               for c in centroids[:i]], axis=0)
            probabilities = distances / np.sum(distances)
            centroids[i] = X[np.random.choice(len(X), p=probabilities)]

        return centroids
