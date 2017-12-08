import numpy as np
from npann.Utilities.misc_functions import accuracy, accuracyBinary
from Sequential import Sequential

class SequentialReinforcement(Sequential):

    def iterate(self, X, y, getReward):
        output = self.forward(X)
        self.loss = self.loss_layer.calculateLoss(output, y)
        curr_grad = self.loss_layer.calculateGrad(output, y)

        reward = getReward(self, X, y)
        for layer in self.layers:
            if layer.__class__.__name__ == "Reinforce":
                layer.reward = reward

        final_grad = self.backward(curr_grad)
        self.update()
        return np.linalg.norm(self.loss)

    def train(self, X, y, minibatch_size, number_epochs, getReward, verbose=1):
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
                minibatch_err = self.iterate(minibatch_X, minibatch_y, getReward)
                epoch_err += minibatch_err

            if verbose==1:
                print "Epoch #%d, Error: %.8f" % (i, epoch_err)
            elif verbose==2:
                print "Epoch #%d, Error: %.8f, Accuracy: %.4f" % (i, epoch_err, accuracyBinary(self, X, y))
