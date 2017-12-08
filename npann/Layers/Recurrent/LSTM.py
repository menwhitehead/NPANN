from npann.Utilities.misc_functions import *
from ..Layer import Layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    # s = sigmoid(x)  # :P  ????
    s = x
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.power(x, 2)

class LSTM(Layer):

    def __init__(self,
                 sequence_length,
                 number_incoming,
                 number_outgoing,
                 weight_init='glorot_uniform'
                 ):
        self.sequence_length = sequence_length
        self.number_incoming = number_incoming
        self.number_outgoing = number_outgoing
        self.forget_weights = weight_inits[weight_init](number_incoming + number_outgoing, number_outgoing)
        self.candidate_weights = weight_inits[weight_init](number_incoming + number_outgoing, number_outgoing)
        self.input_weights = weight_inits[weight_init](number_incoming + number_outgoing, number_outgoing)
        self.output_weights = weight_inits[weight_init](number_incoming + number_outgoing, number_outgoing)
        self.cell_state = np.zeros(number_outgoing)
        self.reset()

    # Reset all remembered activations and gradients
    def reset(self):
        # All activation and gradients are remembered for multiple passes
        self.incoming_acts = []

        self.forget_acts = []
        self.input_acts = []
        self.output_acts = []
        self.candidates = []

        self.incoming_grads = []
        self.outgoing_grads = []
        self.prev_activation = np.zeros(self.number_outgoing)

        self.input_weight_update = np.zeros_like(self.input_weights)
        self.forget_weight_update = np.zeros_like(self.forget_weights)
        self.output_weight_update = np.zeros_like(self.output_weights)
        self.candidate_weight_update = np.zeros_like(self.candidate_weights)

        self.forward_count = 0  # How many times forward is called before backwards start

    def forward(self, x, train=True):
        for i in range(self.sequence_length):
            val = x[i]
            combined_input = np.hstack((val, self.prev_activation))
            self.incoming_acts.append(combined_input)

            self.forget_acts.append(sigmoid(combined_input.dot(self.forget_weights)))
            self.input_acts.append(sigmoid(combined_input.dot(self.input_weights)))
            self.candidates.append(tanh(combined_input.dot(self.candidate_weights)))
            self.cell_state = self.forget_acts[-1] * self.cell_state + self.input_acts[-1] * self.candidates[-1]
            self.output_acts.append(sigmoid(combined_input.dot(self.output_weights)))
            self.prev_activation = self.output_acts[-1] * tanh(self.cell_state)

            self.forward_count += 1

        return self.prev_activation

    def backward(self, curr_grad):

        for i in range(self.sequence_length):
            next_grad = curr_grad.dot(self.weights.T)

            # print self.incoming_acts[-1].T.shape, curr_grad.shape
            # print self.number_incoming, self.number_outgoing
            # layer_grad = np.reshape(self.incoming_acts[-1], (1, 2)).T.dot(np.reshape(curr_grad, (1, 1)))
            layer_grad = np.reshape(self.incoming_acts[-1], (1, self.number_incoming+self.number_outgoing)).T.dot(np.reshape(curr_grad, (1, self.number_outgoing)))
            # print layer_grad, self.weight_update, self.weights

            # Add the update on to the weight_update
            self.weight_update += layer_grad

            # Remove the used up activations
            self.outgoing_acts = self.outgoing_acts[:-1]
            self.incoming_acts = self.incoming_acts[:-1]

            # print next_grad
            curr_input_grad, curr_grad = np.split(next_grad, [self.number_incoming])
            # curr_grad = np.array(np.sum(next_grad)/2.0)

        return curr_grad

    def update(self, optimizer):
        ave_update = self.weight_update / self.forward_count
        layer_update = optimizer.getUpdate(self, ave_update)
        self.weights += layer_update
        self.reset()
