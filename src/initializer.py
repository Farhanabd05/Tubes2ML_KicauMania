import numpy as np


def init_zero(shape) -> np.ndarray:
    return np.zeros(shape)


def init_uniform(shape, lower_bound=0.0, upper_bound=1.0, seed=None) -> np.ndarray:
    np.random.seed(seed)
    return np.random.uniform(lower_bound, upper_bound, size=shape)


def init_normal(shape, mean=0.0, variance=1.0, seed=None) -> np.ndarray:
    np.random.seed(seed)
    return np.random.normal(mean, variance ** (1 / 2), size=shape)


def init_xavier(shape, seed=None) -> np.ndarray:
    np.random.seed(seed)
    n_in, n_out = shape[0], shape[1]
    std_dev = np.sqrt(2 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std_dev


def init_he(shape, seed=None) -> np.ndarray:
    np.random.seed(seed)
    n_in, n_out = shape[0], shape[1]
    std_dev = np.sqrt(2 / n_in)
    return np.random.randn(n_in, n_out) * std_dev
