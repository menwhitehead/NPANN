from misc_functions import *
from Layer import Layer

# Global for now...
# learning_rate = 0.9   # XOR
# learning_rate = 0.001 # MNIST
# learning_rate = 0.05  # Cancer
# learning_rate = 0.1  # BBFN

momentum = 0.9
learning_rate = 0.01 

class Dense(Layer):
    
    def __init__(self, number_incoming, number_outgoing, activation='sigmoid', weight_init='glorot_uniform'):
        self.activation_func = activations[activation]
        self.dactivation_func = dactivations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
    
    def forward(self, x):
        self.incoming_acts = x
        self.outgoing_acts = self.activation_func(x.dot(self.weights))
        return self.outgoing_acts
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * self.dactivation_func(self.outgoing_acts)
        
        #self.rmsgrads = 0.9 * self.rmsgrads + 0.1 * np.power(self.incoming_grad, 2)
        
        #return self.outgoing_grad
        return self.outgoing_grad.dot(self.weights.T)
        
    def update(self):
        #self.incoming_grad *= 1.0 / np.sqrt(self.rmsgrads)
        mterm = momentum * self.prev_update
        actgrad = self.incoming_acts.T.dot(self.outgoing_grad)
        self.prev_update =  mterm + actgrad * learning_rate
        self.weights += self.prev_update
