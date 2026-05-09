import numpy as np

class Conv2D:
    def __init__(self, stride: int = 1, padding: str = 'valid', activation=None):
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.kernel = None   
        self.bias = None    
        self.inputs = None
        self.inputs_padded = None

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.kernel = weights[0]  #
        self.bias = weights[1]   

    def _pad_input(self, inputs: np.ndarray) -> np.ndarray:
        if self.padding == 'valid':
            return inputs

        kH, kW = self.kernel.shape[:2]
        s = self.stride
        _, H, W, _ = inputs.shape

        pad_H = max((H - 1) * s + kH - H, 0)
        pad_W = max((W - 1) * s + kW - W, 0)

        pad_top    = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left   = pad_W // 2
        pad_right  = pad_W - pad_left

        return np.pad(inputs,
                      ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode='constant', constant_values=0)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        is_batched = len(inputs.shape) == 4
        if not is_batched:
            inputs = np.expand_dims(inputs, axis=0)

        self.inputs = inputs

        padded = self._pad_input(inputs)
        self.inputs_padded = padded

        N, H_pad, W_pad, C_in = padded.shape
        kH, kW, _, C_out = self.kernel.shape
        s = self.stride

        H_out = (H_pad - kH) // s + 1
        W_out = (W_pad - kW) // s + 1

        out = np.zeros((N, H_out, W_out, C_out))

        # Sliding window
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                w_start = j * s
                patch = padded[:, h_start:h_start + kH, w_start:w_start + kW, :]

                # dot product 
                out[:, i, j, :] = patch.reshape(N, -1) @ self.kernel.reshape(-1, C_out) + self.bias

        if self.activation is not None:
            out = self.activation.forward(out)

        if not is_batched:
            out = out[0]

        return out

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        is_batched = len(d_values.shape) == 4
        if not is_batched:
            d_values = np.expand_dims(d_values, axis=0)

        if self.activation is not None:
            d_values = self.activation.backward(d_values)

        N, H_out, W_out, C_out = d_values.shape
        kH, kW, C_in, _ = self.kernel.shape
        s = self.stride

        padded = self.inputs_padded
        d_padded = np.zeros_like(padded)
        self.d_kernel = np.zeros_like(self.kernel)
        self.d_bias = np.zeros_like(self.bias)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                w_start = j * s

                patch = padded[:, h_start:h_start + kH, w_start:w_start + kW, :]

                dv_ij = d_values[:, i, j, :]  
                self.d_kernel += np.tensordot(
                    patch.reshape(N, -1),
                    dv_ij,
                    axes=([0], [0])
                ).reshape(kH, kW, C_in, C_out)

                # Gradient terhadap bias
                self.d_bias += dv_ij.sum(axis=0)

                # Gradient terhadap input
                d_padded[:, h_start:h_start + kH, w_start:w_start + kW, :] += \
                    (dv_ij @ self.kernel.reshape(-1, C_out).T).reshape(N, kH, kW, C_in)

        if self.padding == 'same':
            _, H_orig, W_orig, _ = self.inputs.shape
            _, H_pad, W_pad, _ = padded.shape
            pad_top  = (H_pad - H_orig) // 2
            pad_left = (W_pad - W_orig) // 2
            d_inputs = d_padded[:, pad_top:pad_top + H_orig, pad_left:pad_left + W_orig, :]
        else:
            d_inputs = d_padded

        if not is_batched:
            d_inputs = d_inputs[0]

        return d_inputs
