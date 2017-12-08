from npann.Utilities.misc_functions import *
from Layer import Layer

class Dense(Layer):

    def __init__(self, number_incoming,
                 number_outgoing,
                 weight_init='glorot_uniform'):
        self.number_incoming = number_incoming
        self.number_outgoing = number_outgoing
        self.weights = weight_inits[weight_init](self.number_incoming, self.number_outgoing)

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.incoming_acts.dot(self.weights)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad.dot(self.weights.T)
        return self.outgoing_grad

    def getLayerDerivatives(self):
        return self.incoming_acts.T.dot(self.incoming_grad)

    def update(self, optimizer):
        layer_update = optimizer.getUpdate(self, self.getLayerDerivatives())
        self.weights += layer_update
