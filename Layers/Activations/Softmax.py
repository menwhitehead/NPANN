from misc_functions import *
from ..Layer import Layer

class Softmax(Layer):

    def softmax(x):
        ex = np.exp(x)
        z = np.sum(ex)
        result = ex / z
        return result

    def dsoftmax(x):
        #s = softmax(x)
        s = x
        return s * (1 - s)

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.softmax(self.incoming_acts)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * self.dsoftmax(self.outgoing_acts)
        return self.outgoing_grad
