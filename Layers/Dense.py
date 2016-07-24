from misc_functions import *
from Layer import Layer

class Dense(Layer):
    
    def __init__(self, number_incoming,
                 number_outgoing,
                 activation='sigmoid',
                 weight_init='glorot_uniform',
                 learning_rate=0.01,
                 momentum=0.9):
        self.activation_func = activations[activation]
        self.dactivation_func = dactivations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def forward(self, x):
        self.incoming_acts = x
        self.outgoing_acts = self.activation_func(x.dot(self.weights))
        return self.outgoing_acts
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * self.dactivation_func(self.outgoing_acts)
        
        #self.rmsgrads = 0.9 * self.rmsgrads + 0.1 * np.power(self.incoming_grad, 2)
        
        #return self.outgoing_grad
        return self.outgoing_grad.dot(self.weights.T)
        
    def update(self):
        #self.incoming_grad *= 1.0 / np.sqrt(self.rmsgrads)
        mterm = self.momentum * self.prev_update
        actgrad = self.incoming_acts.T.dot(self.outgoing_grad)
        self.prev_update =  mterm + actgrad * self.learning_rate
        self.weights += self.prev_update


# Dense layer for recurrent use
# Remembers activations and gradients from multiple passes
# Update averages weight changes from all the gradient passes
class RecurrentDense(Layer):
    
    def __init__(self,
                 number_incoming,
                 number_outgoing,
                 activation='sigmoid',
                 weight_init='glorot_uniform',
                 learning_rate=0.01,
                 momentum=0.9):
        self.activation_func = activations[activation]
        self.dactivation_func = dactivations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reset()

    # Reset all remembered activations and gradients    
    def reset(self):
        # All activation and gradients are remembered for multiple passes
        self.incoming_acts = []
        self.outgoing_acts = []
        self.incoming_grads = []
        self.outgoing_grads = []
        
        self.weight_update = np.zeros_like(self.weights)
        self.forward_count = 0  # How many times forward is called before backwards start
        
    def forward(self, x):
        self.incoming_acts.append(x)
        self.outgoing_acts.append(self.activation_func(x.dot(self.weights)))
        self.forward_count += 1
        return self.outgoing_acts[-1]
    
    def backward(self, incoming_grad):
        self.incoming_grads.append(incoming_grad)
        self.outgoing_grads.append(self.incoming_grads[-1] * self.dactivation_func(self.outgoing_acts[-1]))
        
        # Remove the activations now that the backprop is finished
        self.outgoing_acts = self.outgoing_acts[:-1]
        
        # Calculate the update that would occur with this backward pass
        update = self.incoming_acts[-1].T.dot(self.outgoing_grads[-1])
        
        # Add the update on to the weight_update
        self.weight_update += update
        
        # Remove activations
        self.incoming_acts = self.incoming_acts[:-1]

        return self.outgoing_grads[-1].dot(self.weights.T)
        
    def update(self):
        mterm = self.momentum * self.prev_update
        ave_update = self.weight_update / self.forward_count
        self.prev_update =  mterm + ave_update * self.learning_rate
        self.weights += self.prev_update
        self.reset()
        
        
        
        
        