

class FiniteDifference(Layer):
    
    def __init__(self, number_incoming, number_outgoing, activation='sigmoid', weight_init='glorot_uniform'):
        self.activation_func = activations[activation]
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
        
        self.outgoing_acts = self.activation_func(x.dot(ws))
        
        #print self.outgoing_acts
        
        return self.outgoing_acts
    
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        #self.outgoing_grad = self.incoming_grad * self.dactivation_func(self.outgoing_acts)
        self.outgoing_grad = self.incoming_grad * 1.0     #self.estimated_grad
        
        # UPdate table with incoming grad info
        #reward = 1.0/incoming_grad
        #reward = -incoming_grad
        #reward = incoming_grad
        # print "INCOMING:",
        rewards = np.zeros_like(incoming_grad)
        # print incoming_grad
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

        # print rewards
        #rewards = np.array([rewards]).T
            
        if self.alter_mode:
            prev_deltas = self.altered_table[-1]
            #self.altered_table[-1] = (prev_deltas, incoming_grad)
            self.altered_table[-1] = (prev_deltas, rewards)
        else:
            #prev_deltas = self.sample_table[-1]
            #self.unaltered_table.append(incoming_grad)
            self.unaltered_table.append(rewards)
            # self.unaltered_table.append(np.zeros_like(incoming_grad))
        
        #return self.outgoing_grad.dot(self.weights.T)
        #self.outgoing_grad = np.mean(self.outgoing_grad, axis=0)
        # print self.outgoing_grad
        # print "RETU"
        # print incoming_grad.shape, rewards.shape, self.weights.T.shape
        # print rewards.dot(self.weights.T)
        #return self.outgoing_grad.dot(self.weights.T)
        #return rewards.dot(self.weights.T)
        # print "REWARDS:"
        # print rewards
        #return rewards
        # print self.outgoing_grad.dot(self.weights.T)
        return self.outgoing_grad.dot(self.weights.T)
        #return self.outgoing_grad
        
        
    def update(self):
        # sys.exit()
        #print self.alter_mode, len(self.altered_table), len(self.unaltered_table)
        #self.prev_update = momentum * self.prev_update + (self.incoming_acts.T.dot(self.outgoing_grad)) * learning_rate
        self.alter_mode = not self.alter_mode
        if (len(self.altered_table) == self.max_table_size) and self.alter_mode:
            grad_deltas = []
            theta_deltas = []
            for i in range(len(self.altered_table)):
                altered_deltas, altered_grads = self.altered_table[i]
                unaltered_grads = self.unaltered_table[i]
                grad_delta = altered_grads - unaltered_grads
                # print "GRAD DEL:", grad_delta

                #grad_deltas.append(np.ndarray.flatten(grad_delta))
                #grad_deltas.append(np.sum(grad_delta))
                grad_deltas.append(np.mean(grad_delta))
                #grad_deltas.append(grad_delta)
                theta_deltas.append(np.ndarray.flatten(altered_deltas))
            
            grad_deltas = np.array(grad_deltas)
            theta_deltas = np.array(theta_deltas)
            
            # inter = theta_deltas.T.dot(theta_deltas)
            # left = np.linalg.inv(inter)
            # right = theta_deltas.T.dot(grad_deltas)
            # weight_update = left.dot(right)
            # print theta_deltas
            # print grad_deltas
            weight_update = np.linalg.lstsq(theta_deltas, grad_deltas)[0]
            # print theta_deltas.dot(weight_update)
            
            # sys.exit()
            
            resized_update = np.reshape(weight_update, self.weights.shape)
            #learning_rate = 1
            # print "OLD WEIGHTS:"
            # print self.weights
            
            # print grad_deltas.shape, theta_deltas.shape, self.weights.shape
            # print "altered table:"
            # print self.altered_table
            # print "unaltered table:"
            # print self.unaltered_table
            # print "Grad Dels:"
            # print grad_deltas
            # print "Theta Dels:"
            # print theta_deltas            
            # print "Left:"
            # print left
            # print "right:"
            # print right
            # print "UPDATE:"
            # print resized_update * learning_rate
            # 
            # sys.exit()

            
            self.weights += self.learning_rate * resized_update
            if self.learning_rate > 0.001:
                self.learning_rate *= 0.99
            #print "LR: ", self.learning_rate
