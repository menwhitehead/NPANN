from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Flatten import Flatten
from Layers.Convolution import Convolution
from Layers.Activations.Sigmoid import Sigmoid
from Optimizers.RMSProp import RMSProp
from Losses.MSE import MSE

np.random.seed(42)

def iterate(s, X, y):
    output = s.forward(X)
    loss = s.loss_layer.calculateLoss(output, y)
    curr_grad = s.loss_layer.calculateGrad(output, y)
    final_grad = s.backward(curr_grad)
    #s.update()
    return loss

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
    img_height = 17
    number_filters = 1
    number_classes = 10
    batch_size = 1

    layer = Convolution(img_width, img_height, number_filters)
    s = Sequential()
    s.addLayer(layer)
    s.addLayer(Flatten())
    s.addLayer(Sigmoid())
    s.addLayer(Dense(number_filters*img_width*img_height, number_classes))
    s.addLayer(Sigmoid())
    s.addLoss(MSE())
    s.addOptimizer(RMSProp())

    X = np.random.rand(batch_size, number_filters, img_width, img_height)
    y = np.random.rand(batch_size, number_classes)

    first_error = iterate(s, X, y)
    # layer_grad = layer.incoming_acts.T.dot(layer.incoming_grad)
    layer_grad = layer.getLayerDerivatives()

    epsilon = 1E-4
    layer.weights[0][0][0] += epsilon
    up_error = iterate(s, X, y)
    layer.weights[0][0][0] -= 2*epsilon
    down_error = iterate(s, X, y)

    print "CONVOLUTION LAYER GRAD TEST:"
    print "numeric grad:  %11.8f" % ((up_error - down_error) / (2 * epsilon))[0][0]
    print "Backprop grad: %11.8f" % layer_grad[0][0][0]
    print

if __name__=="__main__":
    testDense()
    testConvolution()
