import numpy as np
import h5py
import random
from npann_functions import *
from npann import *

# get an array with two 1-hot operands (representing integers) packed in
# add 'em and return a 1-hot result
def addThem(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    x = list(x).index(0.9)
    y = list(y).index(0.9)
    z = x + y
    z = convertToOneHot(z, len(packed_operands))
    return z

# get an array with two 1-hot operands (representing integers) packed in
# and add 1 to the first operand
def addOne(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    x = list(x).index(0.9)
    #y = list(y).index(0.9)
    z = x + 1
    z = convertToOneHot(z, len(packed_operands))
    return z

if __name__ == "__main__":
    
    applying = Dense(20, 20)
    hidden = Dense(50, 30)
    output = Dense(30, 20, activation='softmax')
    func = AiboPG2(30, 2, activation='none')
    exp = AiboPG2(30, 2, activation='none')
    function_library = [addThem, addOne]
    
    model = BBFN(applying, hidden, output, func, exp, function_library)
  
    # ann.addLoss(MSE())
    model.addLoss(CategoricalCrossEntropy())


    X, y = loadAddition()
    print X.shape
    print y.shape
    minibatch_size = 1
    epochs_per_chunk = 100
    number_epochs = 50000

    model.train(X, y, minibatch_size, epochs_per_chunk, verbose=False)
    #output = ann.forward(X)

