import numpy as np
import h5py
import math
import random

DATASETS_DIR = "Datasets/"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.power(x, 2)

def dsigmoid(x):
    #s = sigmoid(x)  # :P  ????
    s = x
    return s * (1 - s)

def relu(x):
    return np.maximum(x, 0, x)

def drelu(x):
    d = np.ones_like(x)
    d[x <= 0] = 0
    return d

def softmax(x):
    #print "X:", x
    ex = np.exp(x)
    #z = np.sum(ex, axis=1)
    z = np.sum(ex)
    #z = z.reshape(z.shape[0], 1)
    result = ex / z
    
    # print "SOFTMAXED:"
    # for i in range(len(result)):
    #     for j in range(len(result[i])):
    #         print x[i][j], 
    #         print result[i][j]
    #     print
    return result

def dsoftmax(x):
    #s = softmax(x)
    s = x
    return s * (1 - s)



def glorotUniformWeights(number_incoming, number_outgoing):
    weight_range = math.sqrt(12.0 / (number_incoming + number_outgoing))
    #self.weights = np.random.normal(0, weight_range, (number_incoming, number_outgoing))
    #self.weights = (0.5 - np.random.rand(number_incoming, number_outgoing)) * weight_range
    return weight_range - (2 * weight_range * np.random.rand(number_incoming, number_outgoing))

def glorotNormalWeights(number_incoming, number_outgoing):
    weight_range = math.sqrt(12.0 / (number_incoming + number_outgoing))
    return np.random.normal(0, weight_range, (number_incoming, number_outgoing))

def normalWeights(number_incoming, number_outgoing):
    return np.random.normal(0, 1.0, (number_incoming, number_outgoing))

def zeroWeights(number_incoming, number_outgoing):
    return np.zeros((number_incoming, number_outgoing))


def convertToOneHot(val, size):
    x = np.zeros(size)
    x[val] = 0.9
    return x


def loadXOR():
    X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    y = np.array([[0,1,1,0]]).T
    return X, y


def loadAddition(number_problems=100, max_number=10):
    xs = []
    ys = []
    for i in range(number_problems):
        operand1 = random.randrange(max_number)
        operand2 = random.randrange(max_number)
        xs.append(np.append(convertToOneHot(operand1, max_number), convertToOneHot(operand2, max_number)))
        ys.append(convertToOneHot(operand1 + operand2, max_number * 2))
    X = np.array(xs)
    y = np.array(ys)

    # for i in range(len(X)):
    #     print X[i],
    #     print y[i]

    return X, y



def loadBreastCancer():
    size = 263
    f = h5py.File(DATASETS_DIR + "breast_cancer.hdf5", 'r')
    X = f['data']['data'][:].T[:size]
    # np.array([f['t_train'][:size]]).T
    y = np.array([f['data']['label'][:size]]).T
    
    littles = np.amin(X, axis=0)
    bigs = np.amax(X, axis=0)
    
    X = (X - littles) / (bigs - littles)
    #print X
    
    y = (y + 1) / 2
    
    print "Breast Cancer Dataset LOADED", X.shape, y.shape
    
    return X, y


def loadMNIST():
    size = 50000
    f = h5py.File(DATASETS_DIR + "mnist.hdf5", 'r')
    X = f['x_train'][:size]
    
    maxes = X.max(axis=0)
    for i in range(len(maxes)):
        if maxes[i] == 0:
            maxes[i] = 0.1
    X *= 1/maxes
    
    raw_y = np.array([f['t_train'][:size]]).T
    
    y = []
    for row in raw_y:
        y.append(convertToOneHot(row[0], 10))
    
    y = np.array(y)
    
    print "MNIST Dataset LOADED"
    
    return X, y




activations = {'sigmoid':sigmoid,
               'relu':relu,
               'softmax':softmax,
               'tanh':tanh,
               'none':lambda(x):x}
dactivations = {'sigmoid':dsigmoid,
                'relu':drelu,
                'softmax':lambda(x):x,
                'tanh':dtanh,
                'none':lambda(x):x}
weight_inits = {'glorot_normal':glorotNormalWeights,
                'glorot_uniform':glorotUniformWeights,
                'normal':normalWeights,
                'zeros':zeroWeights}


if __name__ == "__main__":
    x = 0.5 - np.random.rand(10)
    print x
    print relu(x)
    print drelu(x)


    
