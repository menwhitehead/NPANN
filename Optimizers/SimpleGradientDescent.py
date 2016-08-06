from misc_functions import *
from Optimizer import Optimizer


class SimpleGradientDescent(Optimizer):
    
    def __init__(self, 
                 learning_rate=0.01,
                 momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prev_updates = {}
        
    def getUpdate(self, layer, grad):
        layer_name = str(layer)
        
        if layer_name not in self.prev_updates:
            self.prev_updates[layer_name] = 0.0
            
        mterm = self.momentum * self.prev_updates[layer_name]
        layer_update =  mterm + grad * self.learning_rate
        
        self.prev_updates[layer_name] = layer_update
        
        return layer_update
