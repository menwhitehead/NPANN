from misc_functions import *
from ..Layer import Layer
from ..RecurrentDense import RecurrentDense
from ..Activations.Sigmoid import Sigmoid
from ..Activations.Tanh import Tanh

class GRU(Layer):

    def __init__(self,
                 sequence_length,
                 number_incoming,
                 number_hidden, # also the number outgoing
                 backprop_limit=10
                 ):
        self.sequence_length = sequence_length
        self.number_incoming = number_incoming
        self.number_hidden = number_hidden

        self.reset_layer = RecurrentDense(number_incoming + number_hidden, number_hidden)
        self.reset_act_layer = Sigmoid()
        self.update_layer = RecurrentDense(number_incoming + number_hidden, number_hidden)
        self.update_act_layer = Sigmoid()
        self.hidden_layer = RecurrentDense(number_incoming + number_hidden, number_hidden)
        self.hidden_act_layer = Tanh()

        self.backprop_limit = min(backprop_limit, self.sequence_length)
        self.reset()

    # Reset all remembered activations and gradients
    def reset(self):
        # All activation and gradients are remembered for multiple passes
        self.incoming_acts = []
        self.combined_acts = []
        self.outgoing_acts = []

        self.incoming_grads = []
        self.outgoing_grads = []
        self.prev_activation = np.zeros(self.number_hidden)


    def forward(self, x, train=True):
        for i in range(self.sequence_length):
            self.incoming_acts.append(x[i])
            self.combined_acts.append(np.hstack((self.prev_activation, x[i])))

            r = self.reset_layer.forward(self.combined_acts[-1])
            r = self.reset_act_layer.forward(r)

            z = self.update_layer.forward(self.combined_acts[-1])
            z = self.update_act_layer.forward(z)

            resetted_input = np.hstack((r * self.prev_activation, x[i]))

            h_cand = self.hidden_layer.forward(resetted_input)
            h_cand = self.hidden_act_layer.forward(h_cand)

            self.outgoing_acts.append((1 - z) * self.prev_activation + z * h_cand)
            self.prev_activation = self.outgoing_acts[-1]

        return self.prev_activation


    def backward(self, curr_grad):

        for i in range(self.backprop_limit):

            # :()
            # :()
            # :()
            # :()
            # :()

            hidden_act_grad = self.hidden_act_layer.backward(curr_grad)
            hidden_grad = self.hidden_layer.backward(hidden_act_grad)

            # RESET
            reset_act_grad = self.reset_act_layer.backward(curr_grad)
            reset_grad = self.reset_layer.backward(reset_act_grad)

            # UPDATE
            update_act_grad = self.update_act_layer.backward(curr_grad)
            update_grad = self.update_layer.backward(update_act_grad)

            self.removeLastActs()

            curr_grad = reset_grad + update_grad + hidden_grad

        return curr_grad


    def removeLastActs(self):
        # Remove the used up activations
        self.incoming_acts = self.incoming_acts[:-1]
        self.combined_acts = self.combined_acts[:-1]
        self.outgoing_acts = self.outgoing_acts[:-1]


    def update(self, optimizer):
        self.hidden_layer.update(optimizer)
        self.update_layer.update(optimizer)
        self.reset_layer.update(optimizer)
        self.reset()
