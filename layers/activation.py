import numpy as np
class ReLU:
    def __init__(self):
        self.inputs = None
    def forward(self, inputs : np.ndarray):
        self.inputs = inputs
        return np.maximum(inputs, 0)
    def backward(self, d_values : np.ndarray):
        if self.inputs is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        return d_values * (self.inputs > 0)
    
class Sigmoid:
    def __init__(self):
        self.output = None
    def forward(self, inputs : np.ndarray):
        self.output = 1 / (1 + np.exp(-np.clip(inputs, -250, 250))) # clip utk cgh overflow
        return self.output
    def backward(self, d_values : np.ndarray):
        if self.output is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        return d_values * self.output * (1 - self.output)

class Linear:
    def __init__(self):
        self.inputs = None
    def forward(self, inputs : np.ndarray):
        self.inputs = inputs
        return self.inputs
    def backward(self, d_values : np.ndarray):
        if self.inputs is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        return d_values * 1

class Tanh:
    def __init__(self) -> None:
        self.output = None
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output
    def backward(self, d_values):
        if self.output is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        return d_values * (1 - (self.output)**2)
    
class Softmax:
    def __init__(self):
        self.output = None
    def forward(self, inputs):
        inputs_aman = inputs - np.max(inputs, axis=1, keepdims=True) # utk cegah overflow
        expi = np.exp(inputs_aman)
        total_exp = np.sum(expi, axis=1, keepdims=True)
        self.output = expi / total_exp
        return self.output
    def backward(self, d_values):
        if self.output is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        batch_size, n_class = self.output.shape
        jacobian = np.zeros((batch_size, n_class, n_class))
        d_inputs = np.empty_like(d_values)
        for b in range(batch_size):
            o = self.output[b].reshape(-1, 1)
            jacobian[b] = np.diagflat(self.output[b]) - np.dot(o, o.T)
            d_inputs[b] = np.dot(jacobian[b], d_values[b])
        return d_inputs

class LeakyReLU:
    def __init__(self, alpha = 0.01):
        self.inputs = None
        self.alpha = alpha
    def forward(self, inputs : np.ndarray):
        self.inputs = inputs
        return np.where(inputs >= 0, inputs, self.alpha * inputs)
    def backward(self, d_values : np.ndarray):
        if self.inputs is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        return d_values * np.where(self.inputs >= 0, 1, self.alpha) 

class ELU:
    def __init__(self, alpha = 1.0):
        self.inputs = None
        self.output = None
        self.alpha = alpha
    def forward(self, inputs : np.ndarray):
        self.inputs = inputs
        self.output = np.where(inputs >= 0, inputs, self.alpha * (np.exp(inputs) - 1))
        return self.output
    def backward(self, d_values : np.ndarray):
        if self.inputs is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        return d_values * np.where(self.inputs >= 0, 1, self.output + self.alpha) 