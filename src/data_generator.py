import numpy as np
from sklearn.datasets import make_blobs, make_circles


class DataGenerator:
    @staticmethod
    def gaussian_mixture(n_samples: int = 300, centers: int = 3,
                         random_state: int = None) -> np.ndarray:
        """生成高斯混合数据"""
        X, _ = make_blobs(n_samples=n_samples, centers=centers,
                          random_state=random_state)
        return X

    @staticmethod
    def circles(n_samples: int = 300, noise: float = 0.05,
                random_state: int = None) -> np.ndarray:
        """生成环形数据"""
        X, _ = make_circles(n_samples=n_samples, noise=noise,
                            random_state=random_state)
        return X
