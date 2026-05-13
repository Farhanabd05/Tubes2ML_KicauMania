import numpy as np
try:
    from ...dense import DenseLayer
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from dense import DenseLayer

class DenseOutputLayer(DenseLayer):
    def __init__(self, units, vocab_size):
        super().__init__(units, vocab_size, init_method="zero")

    def set_weights(self, W_keras, b_keras):
        self.W = np.array(W_keras)
        self.b = np.array(b_keras).reshape(1, -1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return super().forward(inputs)
