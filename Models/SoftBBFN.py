import numpy as np
import time
from misc_functions import *

# Building block function network
class SoftBBFN:
    
    def __init__(self, hidden, output, function_library, sequence_length=1):
        self.sequence_length = sequence_length
        self.hidden_layer = hidden # layer for combining previous hidden with applying layer's result
        self.output_layer = output # layer for calculating the predicted calculation output 
        
        # list of functions that can be used by this network
        # Each function must take in ... and produce as output ... ???????
        self.function_library = function_library
        
               
    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer
        
        
    # Feedforward the length of the pre-defined sequence
    def forward(self, X):

        #current_exp = 0 # the part of the input expression that is being focused on
        current_hidden_state = np.zeros((X.shape[0], self.hidden_layer.weights.shape[1]))  # ??? maybe wrong size
        
        for s in range(self.sequence_length):

            # print X
            
            # Call the chosen function for each X pattern, for each function
            #function_results = np.zeros((X.shape[0], (1 + len(self.function_library)) * X.shape[1]))  # +1 for input
            function_results = np.zeros((X.shape[0], (len(self.function_library)) * X.shape[1]))  
            for i in range(len(X)):
                pattern_results = np.zeros((len(self.function_library), X.shape[1]))
                for j in range(len(self.function_library)):
                    #function_results[i][j] = self.function_library[j](X[i])
                    pattern_results[j] = self.function_library[j](X[i])
                    
                #pattern_results = np.hstack((X[i], np.hstack(pattern_results)))
                pattern_results = np.hstack(pattern_results)
                print pattern_results
                function_results[i] = pattern_results
            print "FUNC RESULT:", function_results
            
            stacked = np.stack(function_results, axis=0)
            print "STACKED:", stacked.shape
            
            
            # Take the comp representation and send it (along with the previous
            # hidden state value) into the hidden layer to generate a new hidden state
            combined_input = np.hstack((current_hidden_state, stacked))
            print "ComBI:", combined_input.shape

            current_hidden_state = self.hidden_layer.forward(combined_input)
            print "NEW  HIDDEN:", current_hidden_state.shape
    
    
            # Given the hidden state, now generate 3 values:
            #   A calculation output (the network's predicted result of the calculation)
            #   A new function index (the next fuction to be called)
            #   A new expression mask (the part of the input expression to be focused upon) NOT YET IMPLEMENTED
            calc_output = self.output_layer.forward(current_hidden_state)
            # print "CALC OUTPUT:", calc_output
            
            # current_func_output = self.function_layer.forward(current_hidden_state)
            # current_func_index = np.argmax(current_func_output, axis=1)
            
            # for i in range(len(func_output)):
            #     mx = float(np.max(func_output[i]))
            #     ind = list(func_output[i]).index(mx)
            #     current_func_index[i] = ind

            # print "NEW FUNCIND:", current_func_index
            # print current_func_index, 
            # print "FUNC IND OUTPUT:", func_output
            
            # expression_output = self.expression_layer.forward(current_hidden_state)
            # print "EXP OUTPUT:", expression_output

        return calc_output
    
    
    def backward(self, output, target):
        self.loss = self.loss_layer.calculateLoss(output, target)
        curr_grad = self.loss_layer.calculateGrad(output, target)
        
        
        # Do one starting pass (because it's different)
        #  - No incoming gradient from a future hidden layer (there are no future ones!)
        #  - function_layer and expression_layer are not adjusted
        curr_grad = self.output_layer.backward(curr_grad)
        # print "GRAD AFTER OUTPUT:", curr_grad
        
        curr_grad = self.hidden_layer.backward(curr_grad)
        print "GRAD AFTER HIDDEN:", curr_grad.shape
        
        # split the grad to the two incoming lines
        curr_hidden_grad, curr_grad = np.split(curr_grad, [self.hidden_layer.weights.shape[1]], axis=1)
        print "HIDDEN_GRAD:", curr_hidden_grad.shape
        print "CURR_GRAD:", curr_grad.shape
        
        #curr_grad = self.applying_layer.backward(curr_grad)

        # Once the starting pass is done...
        #   The future_hidden_grad should be set
        #   The func_calc_grad should be set
        #   function_layer and expression_layer can now be backpropped
        #   (but their gradients do not go any further!)
        for i in range(self.sequence_length-1):

           
            # update the calculation output layer
            curr_grad = self.output_layer.backward(curr_grad)
            
            # combine the output
            #curr_grad = (curr_grad + func_grad) / 2.0
            
            # print "GRAD AFTER OUTPUT:", curr_grad
            
            # Need to combine curr_grad (from the main line)
            # along with the future_hidden_grad coming in from the hidden state transitions

            # average the gradients???
            #curr_grad = (curr_grad + future_hidden_grad + func_grad) / 3.0
            curr_grad = (curr_grad + curr_hidden_grad) / 2.0
            # print combined_grad
            
            curr_hidden_grad = self.hidden_layer.backward(curr_hidden_grad)
            # print "GRAD AFTER HIDDEN:", curr_grad
            
            # split the grad to the two incoming lines
            curr_hidden_grad, curr_grad = np.split(curr_hidden_grad, [self.hidden_layer.weights.shape[1]], axis=1)
            # print "HIDDEN_GRAD:", hidden_grad
            # print "FUNCCALC_GRAD:", func_calc_grad
            
            # curr_grad = self.applying_layer.backward(func_calc_grad)
            # print "AFTER APPLYING:", func_calc_grad

            # Need to figure out what to do here...
            # func_calc_grad's size doesn't match the function_indexing output

            
        return curr_grad
    
    def update(self):
        self.hidden_layer.update()
        self.output_layer.update()

    def iterate(self, X, y):
        output = self.forward(X)
        final_grad = self.backward(output, y)
        self.update()
        curr_loss = np.linalg.norm(self.loss)
        #print curr_loss
        return curr_loss
    
    def resetLayers(self):
        pass
    
    def train(self, X, y, minibatch_size, number_epochs, verbose=True):
        dataset_size = len(X)
        for i in range(number_epochs):
            start_time = time.time()
            self.resetLayers() # clear out any old tables/state
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
        return 100.0 * (correct / float(len(X)))
        
        
    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"
            
        return result
        
        
        
