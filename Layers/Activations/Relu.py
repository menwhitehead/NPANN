from misc_functions import *
from ..Layer import Layer

class Relu(Layer):

    def forward(self, x):
        self.incoming_acts = x
        self.outgoing_acts = np.maximum(x, 0)
        return self.outgoing_acts
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * np.ceil(np.array(self.outgoing_acts>0)) 
        return self.outgoing_grad
