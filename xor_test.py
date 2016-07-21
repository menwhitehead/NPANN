import numpy as np
import h5py
import random
from npann_functions import *
from npann import *


if __name__ == "__main__":
    ann = Sequential()
    ann.addLayer(Dense(3, 5, activation='sigmoid', weight_init="glorot_normal"))
    ann.addLayer(Dense(5, 1, activation='sigmoid', weight_init="glorot_normal"))
    ann.addLoss(MSE())
    # ann.addOptimizer('sgd')
    # sys.exit(1)
    X, y = loadXOR()    
    minibatch_size = 4
    number_epochs = 2500

    ann.train(X, y, minibatch_size, number_epochs)
    print ann.forward(X)
    print y
