import numpy as np


class LocallyConnected2D:
    def __init__(self, stride: int = 1, activation=None):
        self.stride = stride
        self.activation = activation

        self.kernel = None       
        self.bias = None         
        self.kernel_size = None 

        self.inputs = None

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.kernel = np.array(weights[0])  
        self.bias   = np.array(weights[1])  

        cfg = keras_layer.get_config()
        ks = cfg['kernel_size']
        self.kernel_size = (ks[0], ks[1]) if hasattr(ks, '__len__') else (ks, ks)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        is_batched = len(inputs.shape) == 4
        if not is_batched:
            inputs = np.expand_dims(inputs, axis=0)

        self.inputs = inputs
        N, H, W, C_in = inputs.shape
        kH, kW = self.kernel_size
        s = self.stride
        C_out = self.kernel.shape[-1]

        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1

        out = np.zeros((N, H_out, W_out, C_out))

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                w_start = j * s

                # (N, kH, kW, C_in) -> flatten -> (N, kH*kW*C_in)
                patch = inputs[:, h_start:h_start + kH, w_start:w_start + kW, :]
                patch_flat = patch.reshape(N, -1)

                # Indeks posisi linear di kernel
                pos = i * W_out + j

                out[:, i, j, :] = patch_flat @ self.kernel[pos] + self.bias[pos]

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
        kH, kW = self.kernel_size
        C_in = self.kernel.shape[1] // (kH * kW)
        s = self.stride

        inputs = self.inputs
        d_inputs = np.zeros_like(inputs)
        self.d_kernel = np.zeros_like(self.kernel)
        self.d_bias = np.zeros_like(self.bias)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * s
                w_start = j * s

                patch = inputs[:, h_start:h_start + kH, w_start:w_start + kW, :]
                patch_flat = patch.reshape(N, -1)

                pos = i * W_out + j
                dv_ij = d_values[:, i, j, :] 

                # Gradient terhadap kernel[pos]: (kH*kW*C_in, C_out)
                self.d_kernel[pos] += patch_flat.T @ dv_ij

                # Gradient terhadap bias[pos]: (C_out,)
                self.d_bias[pos] += dv_ij.sum(axis=0)

                # Gradient terhadap input
                d_inputs[:, h_start:h_start + kH, w_start:w_start + kW, :] += \
                    (dv_ij @ self.kernel[pos].T).reshape(N, kH, kW, C_in)

        if not is_batched:
            d_inputs = d_inputs[0]

        return d_inputs
