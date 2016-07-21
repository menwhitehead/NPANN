import numpy as np
from misc_functions import *

# Building block function network
class BBFN:
    
    def __init__(self, applying, hidden, output, func, exp, function_library, sequence_length=1):
        self.sequence_length = sequence_length
        self.applying_layer = applying  # layer for applying chosen function and getting result
        self.hidden_layer = hidden # layer for combining previous hidden with applying layer's result
        self.output_layer = output # layer for calculating the predicted calculation output 
        self.function_layer = func # layer for choosing the next function index
        self.expression_layer = exp # layer for choosing the next part of the expression to focus on
        
        # list of functions that can be used by this network
        # Each function must take in ... and produce as output ... ???????
        self.function_library = function_library
        
        # self.current_func_index = 0  # the function index for the current training step
        # self.current_exp = 0 # the part of the input expression that is being focused on
        # self.current_hidden_state = np.zeros_like(self.hidden_layer)  # ??? maybe wrong size
        
               
    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer
        
        
    # perform the recurrence as an unrolled computation
    def forward(self, X):
        # print X.shape
        current_func_index = np.zeros(len(X), dtype=np.int32)  # the function index for the current training step (one for each minibatch pattern)
        #current_exp = 0 # the part of the input expression that is being focused on
        current_hidden_state = np.zeros((X.shape[0], 30))  # ??? maybe wrong size
        
        for s in range(self.sequence_length):
        
            # Call the chosen function for each X pattern
            function_results = np.zeros((X.shape[0], 20))
            for i in range(len(X)):
                function_results[i] = self.function_library[current_func_index[i]](X[i])
            # print "FUNC RESULT:", function_results
            
            # Take the function results and map them to an intermediate representation using a Dense layer
            comp_result = self.applying_layer.forward(function_results)
            # print "COMP RESULT:", comp_result
            
            # Take the comp representation and send it (along with the previous
            # hidden state value) into the hidden layer to generate a new hidden state
            combined_input = np.hstack((current_hidden_state, comp_result))
            current_hidden_state = self.hidden_layer.forward(combined_input)
            # print "NEW HIDDEN:", current_hidden_state
    
    
            # Given the hidden state, now generate 3 values:
            #   A calculation output (the network's predicted result of the calculation)
            #   A new function index (the next fuction to be called)
            #   A new expression mask (the part of the input expression to be focused upon) NOT YET IMPLEMENTED
            calc_output = self.output_layer.forward(current_hidden_state)
            # print "CALC OUTPUT:", calc_output
            
            func_output = self.function_layer.forward(current_hidden_state)
            current_func_index = np.argmax(func_output, axis=1)
            
            # for i in range(len(func_output)):
            #     mx = float(np.max(func_output[i]))
            #     ind = list(func_output[i]).index(mx)
            #     current_func_index[i] = ind

            # print "NEW FUNC IND", current_func_index
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
        # print "GRAD AFTER HIDDEN:", curr_grad
        
        # split the grad to the two incoming lines
        future_hidden_grad, curr_grad = np.split(curr_grad, [30], axis=1)
        # print "HIDDEN_GRAD:", future_hidden_grad
        # print "FUNCCALC_GRAD:", func_calc_grad
        
        curr_grad = self.applying_layer.backward(curr_grad)
        
        

        
        # Once the starting pass is done...
        #   The future_hidden_grad should be set
        #   The func_calc_grad should be set
        #   function_layer and expression_layer can now be backpropped
        #   (but their gradients do not go any further!)
        for i in range(self.sequence_length-1):

            # print curr_grad
            grad_for_func = np.array([np.mean(curr_grad, axis=1)])
            # print grad_for_func
            grad_for_func = np.repeat(grad_for_func, self.function_layer.weights.shape[1], axis=0).T
            # print grad_for_func
            # sys.exit()

            # Update the function index-choosing layer
            func_grad = self.function_layer.backward(grad_for_func)
           
            # update the calculation output layer
            curr_grad = self.output_layer.backward(curr_grad)
            
            # combine the output
            #curr_grad = (curr_grad + func_grad) / 2.0
            
            # print "GRAD AFTER OUTPUT:", curr_grad
            
            # Need to combine curr_grad (from the main line)
            # along with the future_hidden_grad coming in from the hidden state transitions
            # print curr_grad.shape
            # print future_hidden_grad.shape
            #sys.exit()
            # average the gradients???
            curr_grad = (curr_grad + future_hidden_grad + func_grad) / 3.0
            # print combined_grad
            # sys.exit()
            
            
            
            curr_grad = self.hidden_layer.backward(curr_grad)
            # print "GRAD AFTER HIDDEN:", curr_grad
            
            # split the grad to the two incoming lines
            hidden_grad, func_calc_grad = np.split(curr_grad, [30], axis=1)
            # print "HIDDEN_GRAD:", hidden_grad
            # print "FUNCCALC_GRAD:", func_calc_grad
            
            func_calc_grad = self.applying_layer.backward(func_calc_grad)
            # print "AFTER APPLYING:", func_calc_grad

            # Need to figure out what to do here...
            # func_calc_grad's size doesn't match the function_indexing output

            
        return curr_grad
    
    def update(self):
        self.applying_layer.update()
        self.hidden_layer.update()
        self.output_layer.update()
        #self.function_layer.update()

    def iterate(self, X, y):
        output = self.forward(X)
        final_grad = self.backward(output, y)
        self.update()
        curr_loss = np.linalg.norm(self.loss)
        #print curr_loss
        return curr_loss
    
    def resetLayers(self):
        self.function_layer.reset()
        self.expression_layer.reset()
    
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
                #print "Epoch #%d, Error: %.8f" % (i, epoch_err)
                if i % 5 == 0:
                    acc = self.accuracy(X, y)
                    print "Epoch #%d\tError: %.4f\tAccuracy: %5.1f%%" % (i, epoch_err, acc)
            
            
    def accuracy(self, X, y):
        dataset_size = len(X)
        correct = 0
        for ind in range(dataset_size):
            curr_out = self.forward(X[ind:ind+1])[0]
            # print curr_out
            max_ind = list(curr_out).index(np.max(curr_out))
            tar_ind = list(y[ind]).index(np.max(y[ind]))
            if max_ind == tar_ind:
                correct += 1
        
        return 100.0 * (correct / float(dataset_size))
        
        
    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"
            
        return result
        
        
        
