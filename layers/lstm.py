import numpy as np

class LSTMLayer:
    def __init__(self):
        self.ht = None
        self.ct = None

    def set_w(self, units: int, kernel, recurrent_kernel, bias):
        self.units = units
        
        self.Wxi = kernel[:, :units]
        self.Wxf = kernel[:, units:units*2]
        self.Wxc = kernel[:, units*2:units*3]
        self.Wxo = kernel[:, units*3:]

        self.Whhi = recurrent_kernel[:, :units]
        self.Whhf = recurrent_kernel[:, units:units*2]
        self.Whhc = recurrent_kernel[:, units*2:units*3]
        self.Whho = recurrent_kernel[:, units*3:]

        self.bi = bias[:units]
        self.bf = bias[units:units*2]
        self.bc = bias[units*2:units*3]
        self.bo = bias[units*3:]

        self.ht = np.zeros(units)
        self.ct = np.zeros(units)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, input_t):
        i = self.sigmoid(np.dot(input_t, self.Wxi) + np.dot(self.ht, self.Whhi) + self.bi)
        f = self.sigmoid(np.dot(input_t, self.Wxf) + np.dot(self.ht, self.Whhf) + self.bf)
        o = self.sigmoid(np.dot(input_t, self.Wxo) + np.dot(self.ht, self.Whho) + self.bo)
        c_tilde = np.tanh(np.dot(input_t, self.Wxc) + np.dot(self.ht, self.Whhc) + self.bc)

        self.ct = (f * self.ct) + (i * c_tilde)
        
        self.ht = o * np.tanh(self.ct)

        return self.ht