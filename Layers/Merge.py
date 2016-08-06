import gnumpy as gnp
import numpy as np
from Layer import Layer

class Merge(Layer):
    
    def __init__(self, number_connections):
        self.number_connections = number_connections
    
    def forward(self, list_inputs):
        return gnp.hstack(list_inputs)
    
    def backward(self, grad):
        return grad
    
    def update(self, optimizer):
        pass
    
    def reset(self):
        pass
