import numpy as np

class RNNLayer:
    def __init__(self):
        self.units = None
        self.W = None
        self.U = None
        self.b = None
        self.ht = None

    def set_w(self, units, kernel, recurrent_kernel, bias):
        self.units = units
        self.W = kernel
        self.U = recurrent_kernel
        self.b = bias
        self.ht = np.zeros(self.units)

    def forward(self, x):
        x_w = np.dot(x, self.W)
        h_u = np.dot(self.ht, self.U)
        
        self.ht = np.tanh(x_w + h_u + self.b)
        
        return self.ht