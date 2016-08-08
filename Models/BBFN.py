import numpy as np
import time
from misc_functions import *

# Building block function network
class BBFN:

    def __init__(self,
                applying,
                applying_act,
                hidden,
                hidden_act,
                output,
                output_act,
                func_net,
                exp_net,
                function_library,
                sequence_length=1):
        self.sequence_length = sequence_length
        self.applying_layer = applying  # layer for applying chosen function and getting result
        self.applying_act = applying_act
        self.hidden_layer = hidden # layer for combining previous hidden with applying layer's result
        self.hidden_act = hidden_act
        self.output_layer = output # layer for calculating the predicted calculation output
        self.output_act = output_act
        self.function_net = func_net # layer for choosing the next function index
        self.expression_net = exp_net # layer for choosing the next part of the expression to focus on

        # list of functions that can be used by this network
        # Each function must take in ... and produce as output ... ???????
        self.function_library = function_library

        # self.current_func_index = 0  # the function index for the current training step
        # self.current_exp = 0 # the part of the input expression that is being focused on
        # self.current_hidden_state = np.zeros_like(self.hidden_layer)  # ??? maybe wrong size


    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer

    def addOptimizer(self, optimizer):
        self.optimizer = optimizer

    # Feedforward the length of the pre-defined sequence
    def forward(self, X):
        # the function index for the current training step (one for each minibatch pattern)
        current_func_output = np.zeros((len(X), len(self.function_library))) # These are the function index activations
        current_func_index = np.zeros(len(X), dtype=np.int32)  # These are the actual integer indices

        #current_exp = 0 # the part of the input expression that is being focused on
        current_hidden_state = np.zeros((X.shape[0], self.hidden_layer.weights.shape[1]))  # ??? maybe wrong size

        for s in range(self.sequence_length):
            current_func_output = self.function_net.forward(current_hidden_state)
            current_func_index = np.argmax(current_func_output, axis=1)
            # print "(%d,%s)" % (s, str(current_func_index)),

            # Call the chosen function for each X pattern
            function_results = np.zeros((X.shape[0], X.shape[1]))
            for i in range(len(X)):
                function_results[i] = self.function_library[current_func_index[i]](X[i])

            # Combine the function call result along with the index of the function that was called
            # This makes it so that the network knows which function was called during the last step
            # combined_input = np.hstack((function_results, current_func_output))

            # Take the function results and map them to an intermediate representation using a Dense layer
            comp_result = self.applying_layer.forward(function_results)
            comp_result = self.applying_act.forward(comp_result)

            # Take the comp representation and send it (along with the previous
            # hidden state value) into the hidden layer to generate a new hidden state
            combined_input = np.hstack((current_hidden_state, comp_result))
            current_hidden_state = self.hidden_layer.forward(combined_input)
            current_hidden_state = self.hidden_act.forward(current_hidden_state)

            # Given the hidden state, now generate 3 values:
            #   A calculation output (the network's predicted result of the calculation)
            #   A new function index (the next fuction to be called)
            #   A new expression mask (the part of the input expression to be focused upon) NOT YET IMPLEMENTED
            calc_output = self.output_layer.forward(current_hidden_state)
            calc_output = self.output_act.forward(calc_output)

        return calc_output


    def backward(self, output, target):
        self.loss = self.loss_layer.calculateLoss(output, target)
        curr_grad = self.loss_layer.calculateGrad(output, target)

        # Update the function index-choosing layer
        if np.array_equal(np.round(output), target):
            reward = np.ones((len(target), self.function_net.layers[0].number_outgoing)) #1
        else:
            reward = np.zeros((len(target), self.function_net.layers[0].number_outgoing)) #0

        # print reward
        self.function_net.layers[1].reward = reward
        fake_grad = np.zeros(self.function_net.layers[-1].outgoing_acts.shape)
        func_grad = self.function_net.backward(fake_grad)

        for i in range(self.sequence_length):

            # update the calculation output layer
            curr_grad = self.output_act.backward(curr_grad)
            curr_grad = self.output_layer.backward(curr_grad)

            # If this isn't the first pass, then we have a curr_hidden_grad set
            if i > 0:
                # average the gradients???
                curr_grad = (curr_grad + curr_hidden_grad) / 2.0

            curr_hidden_grad = self.hidden_act.backward(curr_grad)
            curr_hidden_grad = self.hidden_layer.backward(curr_hidden_grad)

            # split the grad to the two incoming lines
            curr_hidden_grad, func_calc_grad = np.split(curr_hidden_grad, [self.hidden_layer.weights.shape[1]], axis=1)

            curr_grad = self.applying_act.backward(func_calc_grad)
            curr_grad = self.applying_layer.backward(curr_grad)

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
