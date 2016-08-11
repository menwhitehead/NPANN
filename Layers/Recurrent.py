from misc_functions import *
from Dense import Dense

class RecurrentDense(Dense):

    def __init__(self, number_incoming,
                 number_outgoing,
                 weight_init='glorot_uniform'):
        Dense.__init__(self, number_incoming, number_outgoing, weight_init)
        self.outgoing_acts = None

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.incoming_acts.dot(self.weights)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad.dot(self.weights.T)
        return self.outgoing_grad

    def update(self, optimizer):
        self.layer_grad = self.incoming_acts.T.dot(self.incoming_grad)
        # self.layer_grad /= len(self.incoming_acts)
        layer_update = optimizer.getUpdate(self, self.layer_grad)
        self.weights += layer_update
