from misc_functions import *
from Dense import Dense


# Output of this layer is remembered each forward pass
# and then fed in along with the normal input on the next
# forward pass
class Recurrent(Dense):


    def forward(self, x, train=True):
        self.incoming_acts.append(x)
        self.outgoing_acts.append(x.dot(self.weights))
        self.forward_count += 1
        return self.outgoing_acts[-1]

    def backward(self, incoming_grad):
        self.incoming_grads.append(incoming_grad)
        #self.outgoing_grads.append(self.incoming_grads[-1] * self.dactivation_func(self.outgoing_acts[-1]))
        self.outgoing_grads.append(self.incoming_grads[-1].dot(self.weights.T))

        # Remove the activations now that the backprop is finished
        self.outgoing_acts = self.outgoing_acts[:-1]

        # Calculate the update that would occur with this backward pass
        #update = self.incoming_acts[-1].T.dot(self.outgoing_grads[-1])
        layer_grad = self.incoming_acts[-1].T.dot(self.incoming_grads[-1])

        # Add the update on to the weight_update
        self.weight_update += layer_grad

        # Remove activations
        self.incoming_acts = self.incoming_acts[:-1]

        return self.outgoing_grads[-1]

    def update(self, optimizer):
        ave_update = self.weight_update / self.forward_count
        # self.layer_grad = self.incoming_acts.T.dot(self.incoming_grad)
        layer_update = optimizer.getUpdate(self, ave_update)
        self.weights += layer_update
        self.reset()
