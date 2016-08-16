from misc_functions import *
from ..Layer import Layer

def dtanh(x):
    return 1 - np.power(x, 2)

class SimpleRecurrent2(Layer):

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

        self.input_weights = weight_inits[weight_init](number_incoming, number_hidden)
        self.hidden_weights = weight_inits[weight_init](number_hidden, number_hidden)
        self.output_weights = weight_inits[weight_init](number_hidden, number_outgoing)

        self.backprop_limit = min(backprop_limit, self.sequence_length)
        self.reset()

    # Reset all remembered activations and gradients
    def reset(self):
        # All activation and gradients are remembered for multiple passes
        self.incoming_acts = []
        self.outgoing_acts = []

        self.input_acts = []
        self.hidden_acts = []
        self.output_acts = []

        self.incoming_grads = []
        self.outgoing_grads = []
        self.prev_activation = np.zeros(self.number_hidden)

        self.input_weight_update = np.zeros_like(self.input_weights)
        self.hidden_weight_update = np.zeros_like(self.hidden_weights)
        self.output_weight_update = np.zeros_like(self.output_weights)

        self.forward_count = 0  # How many times forward is called before backwards start

    def forward(self, x, train=True):
        for i in range(self.sequence_length):
            val = x[i]
            #combined_input = np.hstack((val, self.prev_activation))
            self.input_acts.append(val.dot(self.input_weights))

            # Use previous activation
            self.hidden_acts.append(self.prev_activation.dot(self.hidden_weights))

            self.incoming_acts.append(val)
            self.outgoing_acts.append(np.tanh(self.input_acts[-1] + self.hidden_acts[-1]))

            self.output_acts.append(self.outgoing_acts[-1].dot(self.output_weights))

            self.prev_activation = self.outgoing_acts[-1]
            self.forward_count += 1

        return self.output_acts[-1]


    def backward(self, curr_grad):
        output_grad = curr_grad.dot(self.output_weights.T)
        self.output_weight_update += self.output_acts[-1].T.dot(curr_grad)

        # curr_grad = dtanh(curr_grad)

        for i in range(self.backprop_limit):
            # print "BACKING", curr_grad
            curr_grad = output_grad * dtanh(self.outgoing_acts[-1])

            hidden_grad = curr_grad.dot(self.hidden_weights.T)
            input_grad = curr_grad.dot(self.input_weights.T)

            # print curr_grad.shape, hidden_grad.shape, input_grad.shape

            if len(self.outgoing_acts) > 1:
                self.hidden_weight_update += self.outgoing_acts[-2].T.dot(curr_grad)
            else:
                self.hidden_weight_update += np.zeros(self.number_hidden)
            # print self.incoming_acts[-1].shape, curr_grad.shape
            self.input_weight_update += np.reshape(self.incoming_acts[-1], (1, self.number_incoming)).T.dot(np.reshape(curr_grad, (1, self.number_hidden)))

            # Calculate the update that would occur with this backward pass
            #layer_grad = self.incoming_acts[-1].T.dot(curr_grad)
            # print self.incoming_acts[-1].shape, curr_grad.shape
            #reshaped = np.reshape(self.incoming_acts[-1], (2, 1))

            # print self.incoming_acts[-1].T.shape, curr_grad.shape
            # print self.number_incoming, self.number_outgoing
            # layer_grad = np.reshape(self.incoming_acts[-1], (1, 2)).T.dot(np.reshape(curr_grad, (1, 1)))
            # layer_grad = np.reshape(self.incoming_acts[-1], (1, self.number_incoming+self.number_outgoing)).T.dot(np.reshape(curr_grad, (1, self.number_outgoing)))

            # Add the update on to the weight_update
            # self.weight_update += layer_grad

            # Remove the used up activations
            self.outgoing_acts = self.outgoing_acts[:-1]
            self.incoming_acts = self.incoming_acts[:-1]
            self.input_acts = self.input_acts[:-1]
            self.hidden_acts = self.hidden_acts[:-1]
            self.output_acts = self.output_acts[:-1]

            # print next_grad
            # curr_input_grad, curr_grad = np.split(next_grad, [self.number_incoming])
            # curr_grad = np.array(np.sum(next_grad)/2.0)
            curr_grad = hidden_grad

        return curr_grad

    def update(self, optimizer):
        # ave_update = self.weight_update / self.forward_count
        ave_update_input = self.input_weight_update / self.backprop_limit
        ave_update_hidden = self.hidden_weight_update / self.backprop_limit
        ave_update_output = self.output_weight_update #/ self.backprop_limit

        # print ave_update_hidden.shape, self.hidden_weights.shape

        # layer_update = optimizer.getUpdate(self, ave_update)
        self.input_weights += optimizer.getUpdate("rnn_input_weights", ave_update_input)
        self.hidden_weights += optimizer.getUpdate("rnn_hidden_weights", ave_update_hidden)
        self.output_weights += optimizer.getUpdate("rnn_output_weights", ave_update_output)
        self.reset()
