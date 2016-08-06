from misc_functions import *
from Layer import Layer

class Dropout(Layer):
    
    def __init__(self, incoming_size, outgoing_size, percentage=0.5):
        self.active = True # changes to False when testing
        self.percentage = percentage
        self.incoming_size = incoming_size
        self.outgoing_size = outgoing_size
    
    def forward(self, x):
        if self.active:
            #self.current_mask = np.random.binomial([np.ones((self.incoming_size, self.outgoing_size))],1-self.percentage)[0]
            self.current_mask = np.random.binomial(1, self.percentage, size=(self.incoming_size, )) / self.percentage
            return self.current_mask * x #x * self.current_mask
        else:
            return x
    
    def backward(self, incoming_grad):
        return self.current_mask * incoming_grad
        
    def update(self):
        pass