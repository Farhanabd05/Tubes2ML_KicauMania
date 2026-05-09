import numpy as np


class MaxPooling2D:
    def __init__(self, pool_size: tuple = (2, 2), strides: tuple = None):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.inputs = None
        self.is_batched = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        # Handle batch or single input
        self.is_batched = len(inputs.shape) == 4
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            
        N, H, W, C = inputs.shape
        pH, pW = self.pool_size
        sH, sW = self.strides
        
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1
        
        out = np.zeros((N, H_out, W_out, C))
        
        # Sliding window
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                h_end = h_start + pH
                w_start = j * sW
                w_end = w_start + pW
                
                # window: (N, pH, pW, C)
                window = inputs[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(window, axis=(1, 2))
                
        if not self.is_batched:
            out = out[0]
            
        return out

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            d_values = np.expand_dims(d_values, axis=0)
            
        N, H, W, C = inputs.shape
        pH, pW = self.pool_size
        sH, sW = self.strides
        
        H_out, W_out = d_values.shape[1:3]
        d_inputs = np.zeros_like(inputs, dtype=np.float64)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                h_end = h_start + pH
                w_start = j * sW
                w_end = w_start + pW
                
                window = inputs[:, h_start:h_end, w_start:w_end, :]
                
                max_val = np.max(window, axis=(1, 2), keepdims=True)
                mask = (window == max_val)
                
                mask_sum = np.sum(mask, axis=(1, 2), keepdims=True)
                grad_route = mask / mask_sum
                
                d_inputs[:, h_start:h_end, w_start:w_end, :] += grad_route * d_values[:, i:i+1, j:j+1, :]
                
        if not self.is_batched:
            d_inputs = d_inputs[0]
            
        return d_inputs


class AveragePooling2D:
    def __init__(self, pool_size: tuple = (2, 2), strides: tuple = None):
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.inputs = None
        self.is_batched = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        
        self.is_batched = len(inputs.shape) == 4
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            
        N, H, W, C = inputs.shape
        pH, pW = self.pool_size
        sH, sW = self.strides
        
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1
        
        out = np.zeros((N, H_out, W_out, C))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                h_end = h_start + pH
                w_start = j * sW
                w_end = w_start + pW
                
                window = inputs[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.mean(window, axis=(1, 2))
                
        if not self.is_batched:
            out = out[0]
            
        return out

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            d_values = np.expand_dims(d_values, axis=0)
            
        N, H, W, C = inputs.shape
        pH, pW = self.pool_size
        sH, sW = self.strides
        
        H_out, W_out = d_values.shape[1:3]
        d_inputs = np.zeros_like(inputs, dtype=np.float64)
        
        average_gradient = d_values / (pH * pW)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * sH
                h_end = h_start + pH
                w_start = j * sW
                w_end = w_start + pW
                
                d_inputs[:, h_start:h_end, w_start:w_end, :] += average_gradient[:, i:i+1, j:j+1, :]
                
        if not self.is_batched:
            d_inputs = d_inputs[0]
            
        return d_inputs


class GlobalMaxPooling2D:

    def __init__(self):
        self.inputs = None
        self.is_batched = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.is_batched = len(inputs.shape) == 4
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            
        out = np.max(inputs, axis=(1, 2))
        
        if not self.is_batched:
            out = out[0]
        return out

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            d_values = np.expand_dims(d_values, axis=0)
            
        max_val = np.max(inputs, axis=(1, 2), keepdims=True)
        mask = (inputs == max_val)
        
        mask_sum = np.sum(mask, axis=(1, 2), keepdims=True)
        grad_route = mask / mask_sum
        
        # Shape d_values dari (N, C) diubah ke (N, 1, 1, C) agar bisa di-broadcast
        d_values_reshaped = np.expand_dims(np.expand_dims(d_values, axis=1), axis=1)
        
        d_inputs = grad_route * d_values_reshaped
        
        if not self.is_batched:
            d_inputs = d_inputs[0]
        return d_inputs


class GlobalAveragePooling2D:

    def __init__(self):
        self.inputs = None
        self.is_batched = False

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.is_batched = len(inputs.shape) == 4
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            
        out = np.mean(inputs, axis=(1, 2))
        
        if not self.is_batched:
            out = out[0]
        return out

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        inputs = self.inputs
        if not self.is_batched:
            inputs = np.expand_dims(inputs, axis=0)
            d_values = np.expand_dims(d_values, axis=0)
            
        N, H, W, C = inputs.shape
        
        d_values_reshaped = np.expand_dims(np.expand_dims(d_values, axis=1), axis=1)
        
        d_inputs = np.ones_like(inputs, dtype=np.float64) * (d_values_reshaped / (H * W))
        
        if not self.is_batched:
            d_inputs = d_inputs[0]
        return d_inputs
