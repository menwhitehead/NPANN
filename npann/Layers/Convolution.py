from npann.Utilities.misc_functions import *
from scipy.signal import convolve2d, correlate2d
from Layer import Layer

class Convolution(Layer):

    def __init__(self,
                 number_filters,
                 incoming_width,
                 incoming_height,
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

    def convolve(self, imgs, filters):
        outputs = []
        for img in imgs:
            filter_outputs = []
            for filter_number in range(len(filters)):
                conv_filter = filters[filter_number]
                curr_filters = np.zeros((self.incoming_height, self.incoming_width))
                for channel_number in range(len(img)):
                    # USE CORRELATE2D and NOT CONVOLVE2D!!!!!
                    curr_filters += correlate2d(img[channel_number], conv_filter, mode="same")
                filter_outputs.append(curr_filters)
            outputs.append(filter_outputs)
        outputs = np.array(outputs)
        return outputs

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.convolve(x, self.weights)
        return self.outgoing_acts

    #### THIS HASN'T BEEN ERROR-CHECKED YET!!!!
    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        roted = np.rot90(np.copy(self.weights), 2)
        self.outgoing_grad = self.convolve(self.incoming_grad, roted)
        return self.outgoing_grad

    def getLayerDerivatives(self):
        weights_grad = np.zeros_like(self.weights)
        for img_index in range(self.incoming_acts.shape[0]):
            for channel_index in range(self.incoming_acts.shape[1]):
                padded = np.pad(np.copy(self.incoming_acts[img_index, channel_index]), 1, mode="constant")
                for filter_index in range(self.incoming_grad.shape[1]):
                    weights_grad[filter_index] += correlate2d(padded, self.incoming_grad[img_index, filter_index], mode="valid")
        return weights_grad

    def update(self, optimizer):
        layer_grad = self.getLayerDerivatives()
        layer_update = optimizer.getUpdate("convfilter", layer_grad)
        self.weights += layer_update
