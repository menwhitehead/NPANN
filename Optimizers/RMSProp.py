from misc_functions import *
from Optimizer import Optimizer


class RMSProp(Optimizer):
    
    def __init__(self, learning_rate=0.01, gamma=0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.prev_updates = {}
        self.little = 1E-7  # Don't divide by zero!
        
    def getUpdate(self, layer, grad):
        layer_name = str(layer)
        
        if layer_name not in self.prev_updates:
            self.prev_updates[layer_name] = 0.0
            
        left = self.gamma * self.prev_updates[layer_name]
        right = (1.0 - self.gamma) * np.power(grad, 2)
        self.prev_updates[layer_name] =  left + right
        adjusted_grad = grad / gnp.sqrt(self.little + self.prev_updates[layer_name])
            
        layer_update = self.learning_rate * adjusted_grad
        
        return layer_update
