from misc_functions import *
from Layer import Layer


class Reinforce(Layer):

    def __init__(self, number_connections,
                 std_dev,
                 activation='sigmoid',
                 weight_init='glorot_uniform'):
        self.number_connections = number_connections
        self.std_dev = std_dev
        self.activation_func = activations[activation]
        self.dactivation_func = dactivations[activation]
        self.reward = 1
        self.stored_grad = 0

    def forward(self, x):
        self.incoming_acts = x
        output = np.random.normal(0, self.std_dev, x.shape)
        # self.outgoing_acts = self.activation_func(self.incoming_acts + output)
        self.outgoing_acts = self.incoming_acts + output
        # print self.outgoing_acts.shape
        # print self.outgoing_acts
        return self.outgoing_acts

    def backward(self, incoming_grad):
        #self.incoming_grad = incoming_grad
        #print self.incoming_grad.shape
        # self.outgoing_grad = self.dactivation_func(self.outgoing_acts)
        # self.outgoing_grad = self.incoming_grad * self.dactivation_func(self.outgoing_acts)
        #self.layer_grad = self.incoming_acts.T.dot(self.outgoing_grad)
        #print self.layer_grad.shape

        output = self.outgoing_acts - self.incoming_acts
        # output = self.dactivation_func(self.outgoing_acts)
        # output -= self.incoming_acts
        output /= self.std_dev**2
        output *= 1 * self.reward
        return output

    def update(self, optimizer):
        pass
