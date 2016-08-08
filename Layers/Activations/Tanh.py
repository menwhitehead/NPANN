from misc_functions import *
from ..Layer import Layer

class Tanh(Layer):
    
    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = np.tanh(x)
        return self.outgoing_acts
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * (1 - np.power(self.outgoing_acts, 2))
        return self.outgoing_grad
        
