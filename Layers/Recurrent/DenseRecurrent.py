from misc_functions import *
from ..Layer import Layer


# Dense layer for recurrent use
# Remembers activations and gradients from multiple passes
# Update averages weight changes from all the gradient passes
class DenseRecurrent(Layer):

    def __init__(self,
                 number_incoming,
                 number_outgoing,
                #  weight_init='glorot_uniform'
                weight_init='normal'
                 ):
        self.number_incoming = number_incoming
        self.number_outgoing = number_outgoing
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.reset()


    # Reset all remembered activations and gradients
    def reset(self):
        self.incoming_acts = []
        self.outgoing_acts = []
        self.incoming_grads = []
        self.outgoing_grads = []
        self.weight_update = np.zeros_like(self.weights)
        self.forward_count = 0  # How many times forward is called before backwards start


    def forward(self, x, train=True):
        self.incoming_acts.append(x)
        self.outgoing_acts.append(x.dot(self.weights))
        self.forward_count += 1
        return self.outgoing_acts[-1]


    def backward(self, incoming_grad):
        self.incoming_grads.append(incoming_grad)
        # print self.incoming_grads[-1].shape, self.weights.T.shape
        self.outgoing_grads.append(self.incoming_grads[-1].dot(self.weights.T))

        # Calculate the update that would occur with this backward pass
        # and add it on to the weight update
        self.weight_update += not1DHorizontal(self.incoming_acts[-1]).T.dot(not1DHorizontal(self.incoming_grads[-1]))
        self.removeLastActs()

        return self.outgoing_grads[-1]


    # Remove the used up activations
    def removeLastActs(self):
        self.incoming_acts = self.incoming_acts[:-1]
        self.outgoing_acts = self.outgoing_acts[:-1]


    def update(self, optimizer):
        ave_update = self.weight_update / self.forward_count
        layer_update = optimizer.getUpdate(self, ave_update)
        self.weights += layer_update
        self.reset()
