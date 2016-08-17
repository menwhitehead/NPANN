from misc_functions import *
from Layer import Layer


class ReinforceRecurrent(Layer):

    def __init__(self, number_connections,
                 std_dev,
                 weight_init='glorot_uniform'):
        self.number_connections = number_connections
        self.std_dev = std_dev
        self.end_std_dev = 0.11
        self.std_dev_decay = 0.99999
        self.reward = 1
        self.stored_grad = 0
        self.incoming_acts = []
        self.outgoing_acts = []

    def forward(self, x, train=True):
        if self.std_dev > self.end_std_dev:
            self.std_dev *= self.std_dev_decay
            # print self.std_dev
        self.incoming_acts.append(x)
        if train:
            output = np.random.normal(0, self.std_dev, x.shape)
            self.outgoing_acts.append(self.incoming_acts[-1] + output)
        else:
            self.outgoing_acts.append(self.incoming_acts[-1])
        return self.outgoing_acts[-1]

    def backward(self, incoming_grad):
        output = self.outgoing_acts[-1] - self.incoming_acts[-1]
        self.outgoing_acts = self.outgoing_acts[:-1]
        self.incoming_acts = self.incoming_acts[:-1]
        output /= self.std_dev**2
        output *= self.reward
        return output
    
    def reset(self):
        self.incoming_acts = []
        self.outgoing_acts = []
