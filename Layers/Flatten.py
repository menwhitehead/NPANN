import numpy as np
from Layer import Layer

class Flatten(Layer):

    def __init__(self):
        self.orig_shape = None

    def forward(self, ar, train=True):
        self.orig_shape = ar.shape
        output = np.reshape(ar, (ar.shape[0], -1))
        return output

    def backward(self, grad):
        return np.reshape(grad, self.orig_shape)

    def update(self, optimizer):
        pass

    def reset(self):
        pass
