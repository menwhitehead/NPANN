from misc_functions import *
from ..Layer import Layer

class LeakyRelu(Layer):
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x):
        self.incoming_acts = x
        self.outgoing_acts = np.maximum(x, 0) + self.alpha * np.minimum(x, 0)
        return self.outgoing_acts
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        out = np.copy(self.outgoing_acts)
        # SLOW
        out[out>0] = 1
        out[out<=0] = self.alpha
        self.outgoing_grad = self.incoming_grad * out
        return self.outgoing_grad
        
