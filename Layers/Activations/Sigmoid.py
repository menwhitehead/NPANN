from misc_functions import *
from ..Layer import Layer

class Sigmoid(Layer):
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.sigmoid(x)
        return self.outgoing_acts
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        s = self.sigmoid(self.outgoing_acts)
        self.outgoing_grad = self.incoming_grad * (s * (1 - s))
        return self.outgoing_grad
        
