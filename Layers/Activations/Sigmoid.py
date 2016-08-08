from misc_functions import *
from ..Layer import Layer

class Sigmoid(Layer):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        # s = sigmoid(x)  # :P  ????
        s = x
        return s * (1 - s)

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.sigmoid(x)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * self.dsigmoid(self.outgoing_acts)
        return self.outgoing_grad
