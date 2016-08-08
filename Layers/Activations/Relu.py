from misc_functions import *
from ..Layer import Layer

class Relu(Layer):

    def relu(self, x):
        return np.maximum(x, 0)

    def drelu(self, x):
        # d = np.ones_like(x)
        # d[x <= 0] = 0
        # return d
        return np.ceil(np.array(x>0))
        # return np.array(x>0, dtype=int)

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.relu(self.incoming_acts)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * self.drelu(self.outgoing_acts)
        return self.outgoing_grad
