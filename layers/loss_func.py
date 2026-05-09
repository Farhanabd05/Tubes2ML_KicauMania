import numpy as np
class MSE:
    def __init__(self) -> None:
        pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return np.mean((y_true - y_pred) ** 2)
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        n = y_pred.shape[0]
        return -2 * (y_true - y_pred) / n
    
class BCE:
    def __init__(self) -> None:
        pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) 
                + (1 - y_true) * np.log(1 - y_pred))
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        n = y_pred.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))/ n 

class CCE:
    def __init__(self) -> None:
        pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        n = y_pred.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred / n