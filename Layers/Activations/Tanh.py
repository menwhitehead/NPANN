from misc_functions import *
from ..Layer import Layer

class Tanh(Layer):

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1 - np.power(x, 2)

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.tanh(self.incoming_acts)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * self.dtanh(self.outgoing_acts)
        return self.outgoing_grad
