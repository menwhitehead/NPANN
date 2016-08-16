import numpy as np
import time
from misc_functions import *

# RNN
class GeneralRecurrent:

    def __init__(self, sequence_length=1):
        self.sequence_length = sequence_length
        self.layers = []
        self.output_connections = {}  # Map outputs from layers to other layers
        self.reverse_connections = {}

    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer

    def addOptimizer(self, optimizer):
        self.optimizer = optimizer

    def addLayer(self, layer):
        self.layers.append(layer)

    def addRecurrentConnection(self, layer_id1, layer_id2):
        "Add a recurrent connection between two layers. (Can be the same layer)"
        if layer_id1 not in self.output_connections:
            self.output_connections[layer_id1] = []
        self.output_connections[layer_id1].appen(layer_id2)

        if layer_id2 not in self.reverse_connections:
            self.reverse_connections[layer_id2] = []
        self.reverse_connections[layer_id2].append(layer_id1)

    def forwardOnline(self, X):
        '''Do a single forward pass through the RNN...
        The rest of the input sequence is not yet ready...'''

        #current_exp = 0 # the part of the input expression that is being focused on
        current_hidden_state = np.zeros((X.shape[0], self.hidden_layer.weights.shape[1]))  # ??? maybe wrong size

        layer_outputs = {}
        for s in range(self.sequence_length):
            curr_input = X

            for layer_id in range(len(self.layers)):
                curr_layer = self.layers[layer_id]

                # Check for recurrent inputs
                if layer_id in self.reverse_connections:
                    recurrent_data = []
                    for incoming_layer in self.reverse_connections[layer_id]:
                        if s > 0:  # Not the first pass, so there is recurrent data available
                            # There is a previously generated layer output that's going to this layer
                            recurrent_data.append(layer_outputs[incoming_layer])
                        else:
                            recurrent_data.append('')
                    # Combine recurrent data with the normal input
                    curr_input = np.hstack(curr_input + recurrent_data)

                # Feed forward (either the input alone or with the recurrent data stacked on)
                curr_input = curr_layer.forward(curr_input)

                # Check if this layer is used for recurrent connections
                # If so, then remember its output for this step
                if layer_id in self.output_connections:
                    layer_outputs[layer_id] = curr_input

        return curr_input


    def forwardFullSequence(self, X):
        pass


    def backward(self, curr_grad):

        layer_outputs = {}
        for s in range(self.sequence_length):
            for layer_id in range(len(self.layers)-1, -1, -1):
                curr_layer = self.layers[layer_id]

                ##################################################NOT COMPLETE

                # Check for recurrent inputs
                if layer_id in self.reverse_connections:
                    recurrent_data = []
                    for incoming_layer in self.reverse_connections[layer_id]:
                        if s > 0:  # Not the first pass, so there is recurrent data available
                            # There is a previously generated layer output that's going to this layer
                            recurrent_data.append(layer_outputs[incoming_layer])
                        else:
                            recurrent_data.append('')
                    # Combine recurrent data with the normal input
                    curr_input = np.hstack(curr_input + recurrent_data)

                # Feed forward (either the input alone or with the recurrent data stacked on)
                curr_input = curr_layer.forward(curr_input)

                # Check if this layer is used for recurrent connections
                # If so, then remember its output for this step
                if layer_id in self.output_connections:
                    layer_outputs[layer_id] = curr_input

        return curr_grad


    def update(self):
        self.applying_layer.update(self.optimizer)
        self.hidden_layer.update(self.optimizer)
        self.output_layer.update(self.optimizer)
        self.function_net.update()

    def iterate(self, X, y):
        output = self.forward(X)
        final_grad = self.backward(output, y)
        self.update()
        curr_loss = np.linalg.norm(self.loss)
        #print curr_loss
        return curr_loss

    # def resetLayers(self):
    #     self.function_layer.reset()
    #     self.expression_layer.reset()

    def train(self, X, y, minibatch_size, number_epochs, verbose=True):
        dataset_size = len(X)
        for i in range(number_epochs):
            start_time = time.time()
            # self.resetLayers() # clear out any old tables/state
            all_minibatch_indexes = np.random.permutation(dataset_size)
            epoch_err = 0
            for j in range(dataset_size / minibatch_size):
                minibatch_indexes = all_minibatch_indexes[j * minibatch_size:j * minibatch_size + minibatch_size]
                minibatch_X = X[minibatch_indexes]
                minibatch_y = y[minibatch_indexes]
                minibatch_err = self.iterate(minibatch_X, minibatch_y)
                epoch_err += minibatch_err
            if verbose:
                # print "Epoch #%d, Error: %.8f in %.2f seconds" % (i, epoch_err, end_time-start_time)
                #if i % 1 == 0:
                    acc = self.accuracy(X, y)
                    end_time = time.time()

                    print "Epoch #%d\tError: %.4f\tAccuracy: %5.1f%% in %.2f seconds" % (i, epoch_err, acc, end_time-start_time)
                    # print "\tAccuracy: %5.1f%%" % (acc)


    def accuracy2(self, X, y):
        outputs = self.forward(X)
        max_outputs = np.argmax(outputs, axis=1)
        max_targets = np.argmax(y, axis=1)
        correct = np.sum(max_targets == max_outputs)
        return 100.0 * (correct / float(len(X)))


    def accuracy(self, X, y, sample_size=100):

        outputs = self.forward(X)
        rounded_outputs = np.round(outputs)
        correct = 0
        for i in range(sample_size):
            # print rounded_outputs[i], y[i]
            if np.array_equal(rounded_outputs[i], y[i]):
                correct += 1
        return 100.0 * (correct / float(sample_size))


    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"

        return result
