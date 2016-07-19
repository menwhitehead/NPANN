import numpy as np
import h5py
import random
from npann_functions import *
from npann import *


if __name__ == "__main__":
    
    ann = Sequential()
    #ann.addLayer(Dense(3, 4, 'sigmoid'))
    ann.addLayer(Dense(784, 10, 'sigmoid'))
    #ann.addLayer(Dense(4, 1, 'sigmoid'))
    ann.addLayer(Dense(10, 10, 'softmax'))
    ann.addLoss(CategoricalCrossEntropy())
    #ann.addLoss(MAE())
    
    X, y = loadMNIST()
    dataset_size = len(X)
    
    minibatch_size = 32

    
    for i in range(10000):
        all_minibatch_indexes = np.random.permutation(dataset_size)
        epoch_err = 0
        for j in range(dataset_size / minibatch_size):
            minibatch_indexes = all_minibatch_indexes[j * minibatch_size:j * minibatch_size + minibatch_size]
            minibatch_X = X[minibatch_indexes]
            minibatch_y = y[minibatch_indexes]
            minibatch_err = ann.iterate(minibatch_X, minibatch_y)
            epoch_err += minibatch_err
            
        print "Epoch #%d, Error: %.8f" % (i, epoch_err)
        random_ind = random.randrange(0, dataset_size)
        
        
        if i % 10 == 0:
            correct = 0
            output = ann.forward(X)
            for ind in range(dataset_size):
                #output = ann.forward(X[ind])
                #print output, y[ind]
                curr_out = output[ind]
                max_ind = list(curr_out).index(np.max(curr_out))
                tar_ind = list(y[ind]).index(np.max(y[ind]))
                if max_ind == tar_ind:
                    correct += 1
            
            print "\t*** Accuracy: %.4f ***" % (correct / float(dataset_size))
            

    
    
    
