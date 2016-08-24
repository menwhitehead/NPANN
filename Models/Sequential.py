import numpy as np
from misc_functions import accuracy, accuracyBinary

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

    def forward(self, X, train=True):
        curr_x = X
        for i in range(len(self.layers)):
            curr_x = self.layers[i].forward(curr_x, train)
        return curr_x

    def backward(self, curr_grad):
        for i in range(len(self.layers)-1, -1, -1):
            curr_grad = self.layers[i].backward(curr_grad)

        return curr_grad

    def update(self):
        for layer in self.layers:
            layer.update(self.optimizer)

    def iterate(self, X, y):
        output = self.forward(X)
        self.loss = self.loss_layer.calculateLoss(output, y)
        curr_grad = self.loss_layer.calculateGrad(output, y)
        final_grad = self.backward(curr_grad)
        self.update()
        return np.linalg.norm(self.loss)

    def resetLayers(self):
        for layer in self.layers:
            layer.reset()

    def train(self, X, y, minibatch_size, number_epochs, verbose=1):
        self.resetLayers()  # clear out any old tables/state
        dataset_size = len(X)
        for i in range(number_epochs):
            all_minibatch_indexes = np.random.permutation(dataset_size)
            epoch_err = 0
            for j in range(dataset_size / minibatch_size):
                minibatch_start = j * minibatch_size
                minibatch_end = (j + 1) * minibatch_size
                minibatch_indexes = all_minibatch_indexes[minibatch_start:minibatch_end]
                minibatch_X = X[minibatch_indexes]
                minibatch_y = y[minibatch_indexes]
                minibatch_err = self.iterate(minibatch_X, minibatch_y)
                epoch_err += minibatch_err

            if verbose==1:
                print "Epoch #%d, Error: %8.4f" % (i, epoch_err)
            elif verbose==2:
                print "Epoch #%d, Error: %8.4f, Accuracy: %6.2f%%" % (i, epoch_err, accuracyBinary(self, X, y))


    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"

        return result
