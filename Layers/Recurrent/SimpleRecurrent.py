from misc_functions import *
from ..Layer import Layer
from DenseRecurrent import DenseRecurrent
from ..Activations.Tanh import Tanh

class SimpleRecurrent(Layer):

    def __init__(self,
                 sequence_length,
                 number_incoming,
                 number_outgoing,
                 weight_init='glorot_uniform'
                 ):
        self.sequence_length = sequence_length
        self.number_incoming = number_incoming
        self.number_outgoing = number_outgoing

        self.input_layer = DenseRecurrent(number_incoming, number_outgoing)
        self.input_layer_act = Tanh()

        self.hidden_layer = DenseRecurrent(number_outgoing, number_outgoing)
        self.hidden_layer_act = Tanh()

        self.backprop_limit = min(10, self.sequence_length)
        self.reset()

    def reset(self):
        self.prev_activation = np.zeros(self.number_outgoing)

    def forward(self, x, train=True):
        for i in range(self.sequence_length):
            input_act = self.input_layer.forward(x[i])
            input_act = self.input_layer_act.forward(input_act)
            hidden_act = self.hidden_layer.forward(self.prev_activation)
            hidden_act = self.hidden_layer_act.forward(hidden_act)
            self.prev_activation = input_act + hidden_act
        return self.prev_activation

    def backward(self, curr_grad):
        for i in range(self.backprop_limit):
            input_grad = self.input_layer_act.backward(curr_grad)
            input_grad = self.input_layer.backward(input_grad)

            hidden_grad = self.hidden_layer_act.backward(curr_grad)
            curr_grad = self.hidden_layer.backward(hidden_grad)

        return curr_grad

    def update(self, optimizer):
        self.input_layer.update(optimizer)
        self.hidden_layer.update(optimizer)
        self.reset()
