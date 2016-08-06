import numpy as np

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
            layer.update(self.optimizer)

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
        
        
    def accuracyBinary(self, X, y):
        outputs = self.forward(X)
        outputs = np.round(outputs)
        correct = np.sum(y == outputs)
        return 100.0 * (correct / float(len(X)))
        
    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"
            
        return result
    
    