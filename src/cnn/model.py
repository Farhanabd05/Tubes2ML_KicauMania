import numpy as np
from typing import List


class CNNModel:
    def __init__(self):
        self.layers: List = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def load_weights_from_keras(self, keras_model) -> None:
        keras_weighted = [l for l in keras_model.layers if len(l.get_weights()) > 0]
        scratch_weighted = [l for l in self.layers if hasattr(l, 'load_weights')]

        if len(keras_weighted) != len(scratch_weighted):
            raise ValueError(
                f"Jumlah weighted layer tidak cocok: "
                f"Keras={len(keras_weighted)}, Scratch={len(scratch_weighted)}"
            )

        for keras_layer, scratch_layer in zip(keras_weighted, scratch_weighted):
            scratch_layer.load_weights(keras_layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        probs = self.forward(inputs)
        return np.argmax(probs, axis=-1)
