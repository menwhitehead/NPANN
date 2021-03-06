# from misc_functions import *
import numpy as np
from npann.Models.Sequential import Sequential
from npann.Layers.Dense import Dense
from npann.Layers.Flatten import Flatten
from npann.Layers.Convolution import Convolution
from npann.Layers.Activations.Sigmoid import Sigmoid
from npann.Optimizers.RMSProp import RMSProp
from npann.Losses.MSE import MSE

np.random.seed(42)

def iterate(s, X, y):
    output = s.forward(X)
    loss = s.loss_layer.calculateLoss(output, y)
    curr_grad = s.loss_layer.calculateGrad(output, y)
    final_grad = s.backward(curr_grad)
    #s.update()
    return loss

def testSigmoid():
    input_dim = 16
    output_dim = 1
    batch_size = 1

    layer = Sigmoid()

    X = np.random.rand(batch_size, input_dim)
    y = np.random.rand(batch_size, output_dim)

    epsilon = 1E-4
    up_error = layer.forward(X + epsilon)
    down_error = layer.forward(X - epsilon)

    print "SIGMOID LAYER GRAD TEST:"
    print "numeric grad:  %11.8f" % ((up_error - down_error) / (2 * epsilon))[0][0]
    print "Backprop grad: %11.8f" % layer.backward(1)[0][0]
    print

def testDense():
    input_dim = 8
    output_dim = 1
    batch_size = 1

    layer = Dense(input_dim, output_dim)
    s = Sequential()
    s.addLayer(layer)
    s.addLayer(Sigmoid())
    s.addLoss(MSE())
    s.addOptimizer(RMSProp())

    X = np.random.rand(batch_size, input_dim)
    y = np.random.rand(batch_size, output_dim)

    first_error = iterate(s, X, y)
    layer_grad = layer.incoming_acts.T.dot(layer.incoming_grad)

    epsilon = 1E-4
    layer.weights[0][0] += epsilon
    up_error = iterate(s, X, y)
    layer.weights[0][0] -= 2*epsilon
    down_error = iterate(s, X, y)

    print "DENSE LAYER GRAD TEST:"
    print "numeric grad:  %11.8f" % ((up_error - down_error) / (2 * epsilon))[0][0]
    print "Backprop grad: %11.8f" % layer_grad[0][0]
    print

def testConvolution():
    img_width = 17
    img_height = img_width
    number_channels = 1
    number_filters = 5
    number_classes = 1#number_filters*img_width*img_height
    batch_size = 1
    epsilon = 1E-5

    layer = Convolution(number_filters, img_width, img_height)
    s = Sequential()
    s.addLayer(layer)
    s.addLayer(Flatten())
    s.addLayer(Sigmoid())
    s.addLayer(Dense(number_filters*img_width*img_height, number_classes))
    s.addLayer(Sigmoid())
    s.addLoss(MSE())
    s.addOptimizer(RMSProp())

    X = np.random.rand(batch_size, number_channels, img_width, img_height)
    y = np.random.rand(batch_size, number_classes)

    first_error = iterate(s, X, y)
    layer_grad = layer.getLayerDerivatives()

    print "CONVOLUTION LAYER GRAD TEST:"
    for filter_index in range(number_filters):
        layer.weights[filter_index][0][0] += epsilon
        up_error = iterate(s, X, y)
        layer.weights[filter_index][0][0] -= 2*epsilon
        down_error = iterate(s, X, y)

        print "Filter %d numeric grad:  %11.8f" % (filter_index, ((up_error - down_error) / (2 * epsilon))[0][0])
        print "        Backprop grad:  %11.8f" % layer_grad[filter_index][0][0]
        print

if __name__=="__main__":
    testSigmoid()
    testDense()
    testConvolution()
