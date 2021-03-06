from npann.Utilities.misc_functions import *
from ..Layer import Layer
from DenseRecurrent import DenseRecurrent
from ..Activations.Sigmoid import Sigmoid
from ..Activations.Tanh import Tanh

class Recurrent(Layer):

    def __init__(self,
                 sequence_length,
                 number_incoming,
                 number_hidden,
                 number_outgoing,
                 weight_init='glorot_uniform',
                 backprop_limit=10
                 ):
        self.sequence_length = sequence_length
        self.number_incoming = number_incoming
        self.number_hidden = number_hidden
        self.number_outgoing = number_outgoing

        self.input_layer = DenseRecurrent(number_incoming, number_hidden)
        self.input_act_layer = Tanh()
        self.hidden_layer = DenseRecurrent(number_hidden, number_hidden)
        self.hidden_act_layer = Tanh()
        self.output_layer = DenseRecurrent(number_hidden, number_outgoing)
        self.output_act_layer = Tanh()

        self.backprop_limit = min(backprop_limit, self.sequence_length)
        self.reset()


    # Reset all remembered activations and gradients
    def reset(self):
        self.incoming_acts = []
        self.hidden_acts = []
        self.outgoing_acts = []
        self.incoming_grads = []
        self.outgoing_grads = []
        self.prev_activation = np.zeros(self.number_hidden)


    def forward(self, x, train=True):
        for i in range(self.sequence_length):
            self.incoming_acts.append(x[i])

            input_act = self.input_layer.forward(x[i])
            input_act = self.input_act_layer.forward(input_act)

            hidden_act = self.hidden_layer.forward(self.prev_activation)
            hidden_act = self.hidden_act_layer.forward(hidden_act)

            output_act = self.output_layer.forward(input_act + hidden_act)
            output_act = self.output_act_layer.forward(output_act)

            self.outgoing_acts.append(output_act)
            self.prev_activation = hidden_act

        return output_act


    def backward(self, curr_grad):
        output_grad = self.output_act_layer.backward(curr_grad)
        curr_grad = self.output_layer.backward(output_grad)

        for i in range(self.backprop_limit):

            # input_grad = self.input_act_layer.backward(curr_grad)
            # input_grad = self.input_layer.backward(input_grad)
            #
            # hidden_grad = self.hidden_act_layer.backward(curr_grad)
            # curr_grad = self.hidden_layer.backward(hidden_grad)
            pass

        return curr_grad


    # Remove the used up activations
    def removeLastActs(self):
        self.incoming_acts = self.incoming_acts[:-1]
        self.hidden_acts = self.hidden_acts[:-1]
        self.outgoing_acts = self.outgoing_acts[:-1]


    def update(self, optimizer):
        self.input_layer.update(optimizer)
        self.hidden_layer.update(optimizer)
        self.output_layer.update(optimizer)
        self.reset()
