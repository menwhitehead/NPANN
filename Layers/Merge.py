import numpy as np
from Layer import Layer

class Merge(Layer):
    
    def forward(self, list_inputs):
        return np.hstack(list_inputs)
    
    def backward(self, grad):
        return grad
    
    def update(self):
        pass
    
    def reset(self):
        pass
