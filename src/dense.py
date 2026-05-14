
import numpy as np
from src.initializer import init_normal, init_uniform, init_zero, init_xavier, init_he


class DenseLayer:
    def __init__(self, input_size, neuron_count, init_method="uniform",
                 seed=None, lower_bound=-0.05, upper_bound=0.05, mean=0.0, variance=1.0,
                 l1_lambda=0.0, l2_lambda=0.0):
        self.input_size = input_size
        self.neuron_count = neuron_count
        shape_W = (input_size, neuron_count)
        shape_b = (1, neuron_count)

        if init_method == "zero":
            self.W = init_zero(shape_W)
        elif init_method == "normal":
            self.W = init_normal(shape_W, mean=mean, variance=variance, seed=seed)
        elif init_method == "uniform":
            self.W = init_uniform(shape_W, lower_bound=lower_bound, upper_bound=upper_bound, seed=seed)
        elif init_method == "xavier":
            self.W = init_xavier(shape_W, seed=seed)
        elif init_method == "he":
            self.W = init_he(shape_W, seed=seed)

        self.b = init_zero(shape_b)
        self.inputs = None
        self.d_W = np.zeros(self.W.shape)
        self.d_b = np.zeros(self.b.shape)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Adam state
        self.t = 0
        self.m_W = np.zeros(self.W.shape)
        self.v_W = np.zeros(self.W.shape)
        self.m_b = np.zeros(self.b.shape)
        self.v_b = np.zeros(self.b.shape)


    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.W = weights[0]               
        self.b = weights[1].reshape(1, -1) 
        self.m_W = np.zeros(self.W.shape)
        self.v_W = np.zeros(self.W.shape)
        self.m_b = np.zeros(self.b.shape)
        self.v_b = np.zeros(self.b.shape)
        self.t = 0

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise ValueError("Method forward harus dipanggil sebelum backward")
        self.d_W = np.dot(self.inputs.T, d_values)
        self.d_W += self.l1_lambda * np.sign(self.W)
        self.d_W += 2 * self.l2_lambda * self.W
        self.d_b = np.sum(d_values, axis=0, keepdims=True)
        return np.dot(d_values, self.W.T)

    def update(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer='sgd'):
        if optimizer == 'sgd':
            self.W -= learning_rate * self.d_W
            self.b -= learning_rate * self.d_b
        elif optimizer == 'adam':
            self.t += 1
            self.m_W = beta1 * self.m_W + (1 - beta1) * self.d_W
            self.m_b = beta1 * self.m_b + (1 - beta1) * self.d_b
            self.v_W = beta2 * self.v_W + (1 - beta2) * (self.d_W) ** 2
            self.v_b = beta2 * self.v_b + (1 - beta2) * (self.d_b) ** 2
            m_W_hat = self.m_W / (1 - beta1 ** self.t)
            v_W_hat = self.v_W / (1 - beta2 ** self.t)
            m_b_hat = self.m_b / (1 - beta1 ** self.t)
            v_b_hat = self.v_b / (1 - beta2 ** self.t)
            self.W -= learning_rate * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
            self.b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    def regularization_loss(self):
        return self.l1_lambda * np.sum(np.abs(self.W)) + self.l2_lambda * np.sum(self.W ** 2)
