import numpy as np
import h5py
import random
import math
from npann_functions import *

# np.random.seed(42)
    
# learning_rate = 0.9  # XOR
# learning_rate = 0.001 # MNIST
# learning_rate = 0.05 # Cancer
momentum = 0.9
learning_rate = 0.1 # Cancer REINFORCE


activations = {'sigmoid':sigmoid,
               'relu':relu,
               'softmax':softmax,
               'tanh':tanh,
               'none':lambda(x):x}
dactivations = {'sigmoid':dsigmoid,
                'relu':drelu,
                'softmax':lambda(x):x,
                'tanh':dtanh,
                'none':lambda(x):x}
weight_inits = {'glorot_normal':glorotNormalWeights,
                'glorot_uniform':glorotUniformWeights,
                'normal':normalWeights,
                'zeros':zeroWeights}
    
    
class CategoricalCrossEntropy:
    
    def calculateLoss(self, output, target):
        #return (target * np.log(output) + (1 - target) * np.log(1 - output))
        return target - output
    
    def calculateGrad(self, output, target):
        return target - output
    
    
class MSE:
    
    def calculateLoss(self, output, target):
        return np.power(target - output, 2)
    
    def calculateGrad(self, output, target):
        return target - output
    
# class MAE:
#     
#     def calculateLoss(self, output, target):
#         return target - output
    
    
    
class Layer:
    
    # def optimizerForward(self, x, opt):
    #     self.optimizer = opt
    #     self.forward(x)
    
    def reset(self):
        pass
        


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
        print "Mask:", mask
        deltas = deltas * mask
        print "delts:", deltas
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
    
    def __init__(self, number_incoming, number_outgoing, activation='sigmoid', weight_init='glorot_uniform'):
        self.activation_func = activations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
        
        self.reward_table = []  # store history of past actions and the results
        self.max_table_size = 100
        self.learning_rate = 3  # large seems pretty good
        self.reset()

    def reset(self):
        self.reward_table = []

    
    def forward(self, x):
        self.incoming_acts = x
        ws = np.copy(self.weights)  # Make a fresh copy of the weights
        max_delta = 1.0
        deltas = max_delta - (2 * max_delta * np.random.rand(self.weights.shape[0], self.weights.shape[1]))
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
            n = np.linalg.norm(final_update)
            if n != 0:
                final_update *= self.learning_rate * 1.0/np.linalg.norm(final_update)
            else:
                final_update *= self.learning_rate * 1.0/0.0001

            # do update
            self.weights += self.learning_rate * final_update





class Dense(Layer):
    
    def __init__(self, number_incoming, number_outgoing, activation='sigmoid', weight_init='glorot_uniform'):
        self.activation_func = activations[activation]
        self.dactivation_func = dactivations[activation]
        self.weights = weight_inits[weight_init](number_incoming, number_outgoing)
        self.prev_update = 0.0
    
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
        self.prev_update = momentum * self.prev_update + (self.incoming_acts.T.dot(self.outgoing_grad)) * learning_rate
        self.weights += self.prev_update


class Sequential:
    
    def __init__(self):
        self.layers = []
        
    def addLayer(self, layer):
        self.layers.append(layer)
        
    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer
        
    def addOptimizer(self, opt):
        self.optimizer = opt
        
    def __getitem__(self, ind):
        return self.layers[ind]
        
    def forward(self, X):
        curr_x = X
        for i in range(len(self.layers)):
            curr_x = self.layers[i].forward(curr_x)
        return curr_x
    
    def backward(self, output, target):
        self.loss = self.loss_layer.calculateLoss(output, target)
        curr_grad = self.loss_layer.calculateGrad(output, target)

        for i in range(len(self.layers)-1, -1, -1):
            curr_grad = self.layers[i].backward(curr_grad)
            
        return curr_grad
    
    def update(self):
        for layer in self.layers:
            layer.update()

    def iterate(self, X, y):
        output = self.forward(X)
        final_grad = self.backward(output, y)
        self.update()
        return np.linalg.norm(self.loss)
    
    def resetLayers(self):
        for layer in self.layers:
            layer.reset()
    
    def train(self, X, y, minibatch_size, number_epochs, verbose=True):
        self.resetLayers()  # clear out any old tables/state
        dataset_size = len(X)
        for i in range(number_epochs):
            all_minibatch_indexes = np.random.permutation(dataset_size)
            epoch_err = 0
            for j in range(dataset_size / minibatch_size):
                minibatch_indexes = all_minibatch_indexes[j * minibatch_size:j * minibatch_size + minibatch_size]
                minibatch_X = X[minibatch_indexes]
                minibatch_y = y[minibatch_indexes]
                minibatch_err = self.iterate(minibatch_X, minibatch_y)
                epoch_err += minibatch_err
                
            if verbose:
                print "Epoch #%d, Error: %.8f" % (i, epoch_err)
            
            
    def accuracy(self, X, y):
        dataset_size = len(X)
        correct = 0
        output = self.forward(X)
        for ind in range(dataset_size):
            curr_out = output[ind]
            max_ind = list(curr_out).index(np.max(curr_out))
            tar_ind = list(y[ind]).index(np.max(y[ind]))
            if max_ind == tar_ind:
                correct += 1
        
        print "\t*** Accuracy: %.4f ***" % (correct / float(dataset_size))
        
        
    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"
            
        return result
        
        
        
        
        
class Graph:
    
    def __init__(self):
        self.layers = {}  # each layer has a name
        self.backward_connections = {} # links from key back to value (both layer names)
        self.forward_connections = {} # links from key forward to value (both layer names)
        self.output_layers = {} # names of layers that output from network
        self.input_layers = {} # names of inputs
        
    def addLayer(self, layer_name, layer, inputs, is_output=False):
        self.layers[layer_name] = layer
        self.backward_connections[layer_name] = []
                
        for inp in inputs:
            prev_layer = self.layers[inp]
            if prev_layer not in self.forward_connections:
                self.forward_connections[prev_layer] = []
            self.forward_connections[prev_layer].append(layer_name)
            self.backward_connections[layer_name].append(prev_layer)

        
    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer
        
    def __getitem__(self, layer_name):
        return self.layers[layer_name]
    
    def setInput(self, input_name, X):
        self.input_layers[input_name] = X
        
    def forward(self, named_input_pairs):  # pairs of (name, X) for inputs
        
        for input_name, X in named_input_pairs:
            self.setInput(input_name, X)
            
            
        # KEEP WORKING HERE, YOU GOOFBALL    
            
        
        curr_x = X
        for i in range(len(self.layers)):
            curr_x = self.layers[i].forward(curr_x)
        return curr_x
    
    def backward(self, output, target):
        self.loss = self.loss_layer.calculateLoss(output, target)
        curr_grad = self.loss_layer.calculateGrad(output, target)

        for i in range(len(self.layers)-1, -1, -1):
            curr_grad = self.layers[i].backward(curr_grad)
            
        return curr_grad
    
    def update(self):
        for layer in self.layers:
            layer.update()

    def iterate(self, X, y):
        output = self.forward(X)
        final_grad = self.backward(output, y)
        self.update()
        return np.linalg.norm(self.loss)
    
    def resetLayers(self):
        for layer in self.layers:
            layer.reset()
    
    def train(self, X, y, minibatch_size, number_epochs, verbose=True):
        self.resetLayers()  # clear out any old tables/state
        dataset_size = len(X)
        for i in range(number_epochs):
            all_minibatch_indexes = np.random.permutation(dataset_size)
            epoch_err = 0
            for j in range(dataset_size / minibatch_size):
                minibatch_indexes = all_minibatch_indexes[j * minibatch_size:j * minibatch_size + minibatch_size]
                minibatch_X = X[minibatch_indexes]
                minibatch_y = y[minibatch_indexes]
                minibatch_err = self.iterate(minibatch_X, minibatch_y)
                epoch_err += minibatch_err
                
            if verbose:
                print "Epoch #%d, Error: %.8f" % (i, epoch_err)
            
            
    def accuracy(self, X, y):
        dataset_size = len(X)
        correct = 0
        output = self.forward(X)
        for ind in range(dataset_size):
            curr_out = output[ind]
            max_ind = list(curr_out).index(np.max(curr_out))
            tar_ind = list(y[ind]).index(np.max(y[ind]))
            if max_ind == tar_ind:
                correct += 1
        
        print "\t*** Accuracy: %.4f ***" % (correct / float(dataset_size))
        
        
    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"
            
        return result
        
        
        
