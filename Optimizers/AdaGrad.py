from misc_functions import *
from Optimizer import Optimizer


class AdaGrad(Optimizer):
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.prev_updates = {}
        self.little = 1E-7  # Don't divide by zero!
        
    def getUpdate(self, layer, grad):
        layer_name = str(layer)
        
        if layer_name not in self.prev_updates:
            self.prev_updates[layer_name] = 0.0
            
        self.prev_updates[layer_name] += np.power(grad, 2)
        
        adjusted_grad = grad / (self.little + np.sqrt(self.prev_updates[layer_name]))
            
        layer_update = self.learning_rate * adjusted_grad
        
        return layer_update
