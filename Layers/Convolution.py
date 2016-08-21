from misc_functions import *
from scipy.signal import convolve2d
from Layer import Layer

class Convolution(Layer):

    def __init__(self, incoming_width,
                 incoming_height,
                 number_filters,
                 kernel_size=3,
                 padding=1,
                 weight_init='glorot_uniform'):
        self.incoming_width = incoming_width
        self.incoming_height = incoming_height
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.weights = []
        for i in range(number_filters):
            self.weights.append(weight_inits[weight_init](self.kernel_size, self.kernel_size))
        self.weights = np.array(self.weights)

    def convolve(self, x):
        outputs = []
        for vec in x:
            filter_outputs = []
            for filter_number in range(len(self.weights)):
                conv_filter = self.weights[filter_number]
                filter_outputs.append(convolve2d(vec[0], conv_filter, mode="same"))
            outputs.append(filter_outputs)
        outputs = np.array(outputs)
        return outputs

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.convolve(x)
        # print "OUTGO:", self.outgoing_acts.shape
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        self.outgoing_grad = self.convolve(incoming_grad)
        return self.outgoing_grad

    def update(self, optimizer):
        # for filter_number in range(len(self.weights)):
        #     self.layer_grad = np.mean(self.incoming_acts * self.incoming_grad, axis=(0, 2, 3))
        #     print "layer grad:", self.layer_grad.shape
        #     layer_update = optimizer.getUpdate("convfilter%d" % filter_number, self.layer_grad)
        #     print "layer update:", layer_update.shape
        #     self.weights[filter_number] += layer_update

        self.layer_grad = np.mean(self.incoming_acts * self.incoming_grad, axis=(0, 2, 3))
        # print "layer grad:", self.layer_grad.shape
        layer_update = optimizer.getUpdate("convfilter", self.layer_grad)
        # print "layer update:", layer_update.shape
        for filter_number in range(len(self.weights)):
            self.weights[filter_number] += layer_update[filter_number]
