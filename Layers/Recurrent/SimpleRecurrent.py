from misc_functions import *
from ..Layer import Layer

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
        self.weights = weight_inits[weight_init](number_incoming + number_outgoing, number_outgoing)
        self.reset()

    # Reset all remembered activations and gradients
    def reset(self):
        # All activation and gradients are remembered for multiple passes
        self.incoming_acts = []
        self.outgoing_acts = []
        self.incoming_grads = []
        self.outgoing_grads = []
        self.prev_activation = np.zeros(self.number_outgoing)
        self.weight_update = np.zeros_like(self.weights)
        self.forward_count = 0  # How many times forward is called before backwards start

    def forward(self, x, train=True):
        for i in range(self.sequence_length):
            val = x[i]
            combined_input = np.hstack((val, self.prev_activation))
            self.incoming_acts.append(combined_input)
            self.outgoing_acts.append(combined_input.dot(self.weights))
            self.forward_count += 1
            self.prev_activation = self.outgoing_acts[-1]
        return self.prev_activation

    def backward(self, curr_grad):
        
        for i in range(self.sequence_length):
            next_grad = curr_grad.dot(self.weights.T)
            print curr_grad, next_grad
    
            # Calculate the update that would occur with this backward pass
            #layer_grad = self.incoming_acts[-1].T.dot(curr_grad)
            print self.incoming_acts[-1].shape, curr_grad.shape
            reshaped = np.reshape(self.incoming_acts[-1], (2, 1))
            
            ###### SIZES NOT QUITE RIGHT
            
            layer_grad = np.dot(reshaped, curr_grad)
            print layer_grad, self.weight_update, self.weights

            # Add the update on to the weight_update
            self.weight_update += layer_grad
    
            # Remove the used up activations
            self.outgoing_acts = self.outgoing_acts[:-1]
            self.incoming_acts = self.incoming_acts[:-1]
            
            #curr_hidden_grad, next_grad = np.split(next_grad, [self.number_incoming])
            #print curr_hidden_grad.shape, next_grad.shape
            curr_grad = next_grad

        return curr_grad

    def update(self, optimizer):
        ave_update = self.weight_update / self.forward_count
        layer_update = optimizer.getUpdate(self, ave_update)
        self.weights += layer_update
        self.reset()
