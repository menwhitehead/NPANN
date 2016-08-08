from misc_functions import *
from Layer import Layer


class FiniteDifference(Layer):
    
    def __init__(self, number_incoming, number_outgoing, weight_init='glorot_uniform'):
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        self.altered_table = []  # store history of past actions and the results
        self.unaltered_table = [] # store 
        self.max_table_size = 100
        self.alter_mode = True  # True means to use an altered set of weights...this flips each call
        self.learning_rate = 0.99

    def reset(self):
        self.altered_table = []
        self.unaltered_table = []
    
    def forward(self, x):
        self.incoming_acts = x

        if self.alter_mode:        
            ws = np.copy(self.weights)  # Make a fresh copy of the weights
            max_delta = 1.0
            deltas = max_delta - (2*max_delta * np.random.rand(self.weights.shape[0], self.weights.shape[1]))
            ws += deltas
            self.altered_table.append(deltas)  # add the chosen tiny moves to the history
        else:
            ws = self.weights
        
        self.outgoing_acts = x.dot(ws)
        
        return self.outgoing_acts
    
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad
        
        # UPdate table with incoming grad info
        #reward = 1.0/incoming_grad
        #reward = -incoming_grad
        #reward = incoming_grad
        rewards = np.zeros_like(incoming_grad)
        for g in range(len(incoming_grad)):
            gr = incoming_grad[g]
            if len(gr) > 1:
                for g2 in range(len(gr)):
                    gr2 = gr[g2]
                    if abs(gr2) < 0.5:
                        rewards[g][g2] = 1.0
            else:
                if abs(gr) < 0.5:
                    rewards[g] = 1.0
            # rewards[g] = 1.0 - abs(gr)

        if self.alter_mode:
            prev_deltas = self.altered_table[-1]
            self.altered_table[-1] = (prev_deltas, rewards)
        else:
            self.unaltered_table.append(rewards)

        return self.outgoing_grad.dot(self.weights.T)
        
    def update(self):
        self.alter_mode = not self.alter_mode
        if (len(self.altered_table) == self.max_table_size) and self.alter_mode:
            grad_deltas = []
            theta_deltas = []
            for i in range(len(self.altered_table)):
                altered_deltas, altered_grads = self.altered_table[i]
                unaltered_grads = self.unaltered_table[i]
                grad_delta = altered_grads - unaltered_grads
                grad_deltas.append(np.mean(grad_delta))
                theta_deltas.append(np.ndarray.flatten(altered_deltas))
            
            grad_deltas = np.array(grad_deltas)
            theta_deltas = np.array(theta_deltas)
            
            weight_update = np.linalg.lstsq(theta_deltas, grad_deltas)[0]
            resized_update = np.reshape(weight_update, self.weights.shape)
            
            self.weights += self.learning_rate * resized_update
            if self.learning_rate > 0.001:
                self.learning_rate *= 0.99
