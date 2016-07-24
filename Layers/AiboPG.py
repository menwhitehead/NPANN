from misc_functions import *
from Layer import Layer

class AiboPG(Layer):
    
    def __init__(self, number_incoming, number_outgoing, activation='sigmoid', weight_init='glorot_uniform'):
        self.activation_func = activations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        
        self.reward_table = []  # store history of past actions and the results
        self.max_table_size = 50
        self.learning_rate = 3  # large seems pretty good
        self.reset()

    def reset(self):
        self.reward_table = []
        max_delta = 1.0
        self.deltas = max_delta * np.random.rand(self.weights.shape[0], self.weights.shape[1])
    
    def forward(self, x):
        self.incoming_acts = x
        ws = np.copy(self.weights)  # Make a fresh copy of the weights
        deltas = np.copy(self.deltas)
        mask = np.random.rand(deltas.shape[0], deltas.shape[1])
        # ORDER MATTERS!
        mask[mask < 0.33] = -1
        mask[mask > 0.66] = 0
        mask[mask > 0.33] = 1
        # print "Mask:", mask
        deltas = deltas * mask
        # print "delts:", deltas
        ws += deltas
        self.reward_table.append(deltas)  # add the chosen tiny moves to the history
        self.outgoing_acts = self.activation_func(x.dot(ws))
                
        return self.outgoing_acts
    
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * 1.0     #self.estimated_grad

        rewards = np.zeros_like(incoming_grad)
        for g in range(len(incoming_grad)):
            gr = incoming_grad[g]
            if len(gr) > 1:
                for g2 in range(len(gr)):
                    gr2 = gr[g2]
                    if abs(gr2) < 0.5:
                        rewards[g][g2] = 1.0 - abs(gr2)  #1.0
                    else:
                        rewards[g][g2] = 1.0 - abs(gr2)
            else:
                if abs(gr) < 0.5:
                    rewards[g] = 1.0

        prev_deltas = self.reward_table[-1]
        rewards = np.mean(rewards)
        self.reward_table[-1] = (prev_deltas, rewards)

        return self.outgoing_grad.dot(self.weights.T)
        
        
    def update(self):

        if (len(self.reward_table) == self.max_table_size):
            scores = np.zeros((self.weights.shape[0], self.weights.shape[1], 3))  # for each weight delta, sum up rewards for pos, 0, neg
            for i in range(len(self.reward_table)):
                deltas, reward = self.reward_table[i]
                for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if deltas[j][k] > 0:
                            scores[j][k][0] += reward
                        elif deltas[j][k] == 0:
                            scores[j][k][1] += reward
                        else:
                            scores[j][k][2] += reward
                            
            #average_scores = np.mean(scores, axis=2)
            #print "ave scores:"
            #print average_scores
            
            final_update = np.zeros_like(self.weights)
            for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if scores[j][k][1] > scores[j][k][0] and scores[j][k][1] > scores[j][k][2]:
                            final_update[j][k] = 0.0
                        else:
                            final_update[j][k] = scores[j][k][0] - scores[j][k][2]
                        
                        
            # normalize update
            final_update *= self.learning_rate * 1.0/np.linalg.norm(final_update)

            # do update
            self.weights += self.learning_rate * final_update





class AiboPG2(Layer):
    
    def __init__(self,
                 number_incoming,
                 number_outgoing,
                 activation='sigmoid',
                 weight_init='glorot_uniform',
                 learning_rate=3,
                 max_table_size=100
                 ):
        self.activation_func = activations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        
        self.reward_table = []  # store history of past actions and the results
        self.max_table_size = max_table_size
        self.learning_rate = learning_rate # large seems pretty good
        self.reset()

    def reset(self):
        self.reward_table = []
        # An index into the layer's reward table used to match up deltas and their rewards
        self.curr_index = 0

    
    def forward(self, x):
        ws = np.copy(self.weights)  # Make a fresh copy of the weights
        max_delta = 1.0
        deltas = max_delta - (2 * max_delta * np.random.rand(self.weights.shape[0], self.weights.shape[1]))
        ws += deltas
        self.reward_table.append(deltas)  # add the chosen tiny moves to the history
        # self.curr_index += 1
        return self.activation_func(x.dot(ws))

    
    
    def backward(self, incoming_grad):
        rewards = np.zeros_like(incoming_grad)
        for g in range(len(incoming_grad)):
            gr = incoming_grad[g]
            if len(gr) > 1:
                for g2 in range(len(gr)):
                    gr2 = gr[g2]
                    if abs(gr2) < 0.5:
                        rewards[g][g2] = 1.0 - abs(gr2)  #1.0
                    else:
                        rewards[g][g2] = 1.0 - abs(gr2)
            else:
                if abs(gr) < 0.5:
                    rewards[g] = 1.0

        #prev_deltas = self.reward_table[-1]
        prev_deltas = self.reward_table[self.curr_index]
        rewards = np.mean(rewards)
        self.reward_table[self.curr_index] = (prev_deltas, rewards)
        self.curr_index += 1

        return incoming_grad.dot(self.weights.T)
        
        
    def update(self):
        #print len(self.reward_table)
        if (len(self.reward_table) >= self.max_table_size):
            # for each weight delta, sum up rewards for pos, 0, neg
            scores = np.zeros((self.weights.shape[0], self.weights.shape[1], 3))  
            for i in range(len(self.reward_table)):
                # print i, self.curr_index
                #print self.reward_table[i]
                deltas, reward = self.reward_table[i]
                for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if deltas[j][k] > 0:
                            scores[j][k][0] += reward
                        elif deltas[j][k] == 0:
                            scores[j][k][1] += reward
                        else:
                            scores[j][k][2] += reward
                            
            #average_scores = np.mean(scores, axis=2)
            #print "ave scores:"
            #print average_scores
            
            final_update = np.zeros_like(self.weights)
            for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if scores[j][k][1] > scores[j][k][0] and scores[j][k][1] > scores[j][k][2]:
                            final_update[j][k] = 0.0
                        else:
                            final_update[j][k] = scores[j][k][0] - scores[j][k][2]
                        
                        
            # normalize update
            n = np.linalg.norm(final_update)
            if n != 0:
                final_update *= self.learning_rate * 1.0/np.linalg.norm(final_update)
            else:
                final_update *= self.learning_rate * 1.0/0.0001

            # do update
            self.weights += self.learning_rate * final_update
            self.reset()
            
            


class AiboPGRecurrent(Layer):
    
    def __init__(self,
                 number_incoming,
                 number_outgoing,
                 activation='sigmoid',
                 weight_init='glorot_uniform',
                 learning_rate=3,
                 max_table_size=100
                 ):
        self.activation_func = activations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        self.curr_index = 0
        
        self.reward_table = []  # store history of past actions and the results
        self.max_table_size = max_table_size
        self.learning_rate = learning_rate # large seems pretty good
        self.reset()

    def reset(self):
        self.reward_table = []
        # An index into the layer's reward table used to match up deltas and their rewards
        self.curr_index = 0
        max_delta = 1.0
        self.deltas = max_delta * np.random.rand(self.weights.shape[0], self.weights.shape[1])
    

    def forward(self, x):
        self.incoming_acts = x
        ws = np.copy(self.weights)  # Make a fresh copy of the weights
        deltas = np.copy(self.deltas)
        mask = np.random.rand(deltas.shape[0], deltas.shape[1])
        # ORDER MATTERS!
        mask[mask < 0.33] = -1
        mask[mask > 0.66] = 0
        mask[mask > 0.33] = 1
        # print "Mask:", mask
        deltas = deltas * mask
        # print "delts:", deltas
        ws += deltas
        self.reward_table.append(deltas)  # add the chosen tiny moves to the history
        self.outgoing_acts = self.activation_func(x.dot(ws))
                
        return self.outgoing_acts
    
    
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.incoming_grad * 1.0     #self.estimated_grad

        rewards = np.zeros_like(incoming_grad)
        for g in range(len(incoming_grad)):
            gr = incoming_grad[g]
            if len(gr) > 1:
                for g2 in range(len(gr)):
                    gr2 = gr[g2]
                    if abs(gr2) < 0.5:
                        rewards[g][g2] = 1.0 - abs(gr2)  #1.0
                    else:
                        rewards[g][g2] = 1.0 - abs(gr2)
            else:
                if abs(gr) < 0.5:
                    rewards[g] = 1.0

        prev_deltas = self.reward_table[self.curr_index]
        rewards = np.mean(rewards)
        self.reward_table[self.curr_index] = (prev_deltas, rewards)
        self.curr_index += 1
        return self.outgoing_grad.dot(self.weights.T)
        
        
    def update(self):

        if (len(self.reward_table) == self.max_table_size):
            scores = np.zeros((self.weights.shape[0], self.weights.shape[1], 3))  # for each weight delta, sum up rewards for pos, 0, neg
            for i in range(len(self.reward_table)):
                deltas, reward = self.reward_table[i]
                for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if deltas[j][k] > 0:
                            scores[j][k][0] += reward
                        elif deltas[j][k] == 0:
                            scores[j][k][1] += reward
                        else:
                            scores[j][k][2] += reward
                            
            #average_scores = np.mean(scores, axis=2)
            #print "ave scores:"
            #print average_scores
            
            final_update = np.zeros_like(self.weights)
            for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if scores[j][k][1] > scores[j][k][0] and scores[j][k][1] > scores[j][k][2]:
                            final_update[j][k] = 0.0
                        else:
                            final_update[j][k] = scores[j][k][0] - scores[j][k][2]
                        
                        
            # normalize update
            final_update *= self.learning_rate * 1.0/np.linalg.norm(final_update)

            # do update
            self.weights += self.learning_rate * final_update
            self.reset()





            
            
class AiboPG2Recurrent(Layer):
    
    def __init__(self,
                 number_incoming,
                 number_outgoing,
                 activation='sigmoid',
                 weight_init='glorot_uniform',
                 learning_rate=3,
                 max_table_size=100
                 ):
        self.activation_func = activations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        
        self.reward_table = []  # store history of past actions and the results
        self.max_table_size = max_table_size
        self.learning_rate = learning_rate # large seems pretty good
        self.reset()

    def reset(self):
        self.reward_table = []
        # An index into the layer's reward table used to match up deltas and their rewards
        self.curr_index = 0

    
    def forward(self, x):
        ws = np.copy(self.weights)  # Make a fresh copy of the weights
        max_delta = 1.0
        deltas = max_delta - (2 * max_delta * np.random.rand(self.weights.shape[0], self.weights.shape[1]))
        ws += deltas
        self.reward_table.append(deltas)  # add the chosen tiny moves to the history
        # self.curr_index += 1
        return self.activation_func(x.dot(ws))

    
    
    def backward(self, incoming_grad):
        rewards = np.zeros_like(incoming_grad)
        for g in range(len(incoming_grad)):
            gr = incoming_grad[g]
            if len(gr) > 1:
                for g2 in range(len(gr)):
                    gr2 = gr[g2]
                    if abs(gr2) < 0.5:
                        rewards[g][g2] = 1.0 - abs(gr2)  #1.0
                    else:
                        rewards[g][g2] = 1.0 - abs(gr2)
            else:
                if abs(gr) < 0.5:
                    rewards[g] = 1.0

        #prev_deltas = self.reward_table[-1]
        prev_deltas = self.reward_table[self.curr_index]
        rewards = np.mean(rewards)
        self.reward_table[self.curr_index] = (prev_deltas, rewards)
        self.curr_index += 1

        return incoming_grad.dot(self.weights.T)
        
        
    def update(self):
        #print len(self.reward_table)
        if (len(self.reward_table) >= self.max_table_size):
            # for each weight delta, sum up rewards for pos, 0, neg
            scores = np.zeros((self.weights.shape[0], self.weights.shape[1], 3))  
            for i in range(len(self.reward_table)):
                # print i, self.curr_index
                #print self.reward_table[i]
                deltas, reward = self.reward_table[i]
                for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if deltas[j][k] > 0:
                            scores[j][k][0] += reward
                        elif deltas[j][k] == 0:
                            scores[j][k][1] += reward
                        else:
                            scores[j][k][2] += reward
                            
            #average_scores = np.mean(scores, axis=2)
            #print "ave scores:"
            #print average_scores
            
            final_update = np.zeros_like(self.weights)
            for j in range(self.weights.shape[0]):
                    for k in range(self.weights.shape[1]):
                        if scores[j][k][1] > scores[j][k][0] and scores[j][k][1] > scores[j][k][2]:
                            final_update[j][k] = 0.0
                        else:
                            final_update[j][k] = scores[j][k][0] - scores[j][k][2]
                        
                        
            # normalize update
            n = np.linalg.norm(final_update)
            if n != 0:
                final_update *= self.learning_rate * 1.0/np.linalg.norm(final_update)
            else:
                final_update *= self.learning_rate * 1.0/0.0001

            # do update
            self.weights += self.learning_rate * final_update
            self.reset()



