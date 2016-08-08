from misc_functions import *
from Layer import Layer


class Reinforce(Layer):

    def __init__(self, number_connections,
                 std_dev,
                 weight_init='glorot_uniform'):
        self.number_connections = number_connections
        self.std_dev = std_dev
        self.reward = 1
        self.stored_grad = 0

    def forward(self, x, train=True):
        self.incoming_acts = x
        output = np.random.normal(0, self.std_dev, x.shape)
        self.outgoing_acts = self.incoming_acts + output
        return self.outgoing_acts

    def backward(self, incoming_grad):
        output = self.outgoing_acts - self.incoming_acts
        output /= self.std_dev**2
        output *= 1 * self.reward
        return output

