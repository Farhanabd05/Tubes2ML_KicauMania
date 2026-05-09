import numpy as np


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # input shape (H, W, C) single, (N, H, W, C) batch, output shape (H*W*C,) single, (N, H*W*C) batch
        self.input_shape = inputs.shape
        if len(self.input_shape) == 4:
            N = self.input_shape[0]
            return inputs.reshape((N, -1))
        else:
            return inputs.reshape(-1)

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        return d_values.reshape(self.input_shape)
