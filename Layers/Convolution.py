from misc_functions import *
from scipy.signal import convolve2d, correlate2d
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

    def npconvolve(self, imgs, filters):
        outputs = []
        for img in imgs:
            for channel_number in range(len(img)):
                filter_outputs = []
                for filter_number in range(len(filters)):
                    conv_filter = filters[filter_number]
                    #filter_outputs.append(convolve2d(img[channel_number], conv_filter, mode="same"))
                    filter_outputs.append(correlate2d(img[channel_number], conv_filter, mode="same"))
                outputs.append(filter_outputs)
        outputs = np.array(outputs)
        return outputs

    def baseConvolve(self, arr1, arr2):
        print "CONVING :"
        print arr1
        print arr2
        return np.sum(arr1 * arr2)

    def convolve(self, image, filt):
        "Use filt as a convolution filter throughout image"
        print "FILT:", filt
        output = np.zeros_like(image)
        image = np.pad(image, 1, 'constant')
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                print "\t", i, j
                tmp_arr = image[i-1:i+2, j-1:j+2]  # TODO: variable size kernels!
                # assert(tmp_arr.shape == filt.shape)
                convolution = self.baseConvolve(tmp_arr, filt)
                print convolution
                output[i-1, j-1] = convolution
        print "CONV:", output
        return output

    def convolveAll(self, arr1, arr2):
        output = np.zeros((arr1.shape[0], len(arr2), arr1.shape[2], arr1.shape[3]))
        for j in range(len(arr2)):
            output[j] = self.convolve(arr1[0][0], arr2[j])
        return output

    def convolveAllFilters(self, image):
        # output = np.zeros((image.shape[0], self.number_filters, image.shape[2], image.shape[3]))
        # for f in range(len(image)):
        #     output[f] = self.convolve(image, self.weights[f])
        # return output
        return self.convolveAll(image, self.weights)

    def forward(self, x, train=True):
        self.incoming_acts = x
        self.outgoing_acts = self.npconvolve(x, self.weights)
        # self.outgoing_acts = self.convolveAll(x, self.weights)
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        # roted = np.copy(self.incoming_grad)
        #np.rot90(roted, 2)
        # print self.incoming_grad
        # np.rot90(self.incoming_grad, 2)
        # print self.incoming_grad
        roted = np.rot90(np.copy(self.weights), 2)

        self.outgoing_grad = self.npconvolve(self.incoming_grad, roted)
        # print "BACKWARD SHAPES:", self.incoming_grad.shape, self.outgoing_grad.shape
        # print "OUTGOING:", self.outgoing_grad
        return self.outgoing_grad

    
    def getLayerDerivativesWORKING(self):
        result = np.zeros_like(self.weights)
        padded = np.pad(np.copy(self.incoming_acts[0][0]), 1, mode="constant")
        print padded
        print self.incoming_grad
        # padded = np.rot90(padded, 2)
        for center_y in range(self.incoming_grad.shape[2]):
            for center_x in range(self.incoming_grad.shape[3]):
                print center_y, center_x
                g = self.incoming_grad[0, 0, center_y, center_x]
                print "G::", g
                for k1 in range(-1, 2):
                    for k2 in range(-1, 2):
                        #if center_y+k1 >= 0 and center_y+k1 < padded.shape[0]:
                            # if center_x+k2 >= 0 and center_x+k2 < padded.shape[1]:
                                # print "\t", k1+1, k2+1, center_y+k1, center_x+k2
                                inc = padded[center_y+k1+1, center_x+k2+1]
                                print "\t\t", inc
                                result[0, k1+1, k2+1] += g * inc

        print "DERIVES:"
        print result
        return result
    
    def getLayerDerivatives(self):
        filters_grad2 = np.zeros_like(self.weights)
        padded = np.pad(np.copy(self.incoming_acts[0][0]), 1, mode="constant")
        filters_grad2[0] = correlate2d(padded, self.incoming_grad[0][0], mode="valid")
        return filters_grad2
        
    def getLayerDerivativesLoopy(self):
        padded = np.pad(np.copy(self.incoming_acts[0][0]), 1, mode="constant")
        img_h = padded.shape[0]
        img_w = padded.shape[1]
        filters_grad = np.zeros_like(self.weights)
        for y in range(self.incoming_grad.shape[2]):
            for x in range(self.incoming_grad.shape[3]):
                inc = self.incoming_grad[0, 0, y, x]
                chunk = padded[y:y+self.kernel_size, x:x+self.kernel_size]
                filters_grad[0, :, :] += chunk * inc
                
        
                
        return filters_grad
    

    def update(self, optimizer):
        # for filter_number in range(len(self.weights)):
        #     self.layer_grad = np.mean(self.incoming_acts * self.incoming_grad, axis=(0, 2, 3))
        #     print "layer grad:", self.layer_grad
        #     layer_update = optimizer.getUpdate("convfilter%d" % filter_number, self.layer_grad)
        #     # print "layer update:", layer_update.shape
        #     self.weights[filter_number] += layer_update

        layer_grad = self.getLayerDerivatives()

        #### SEEMS FISHY///...../////
        # print "INCOMING GRAD:", self.incoming_grad.shape
        # self.layer_grad = np.mean(self.incoming_grad, axis=(0, 2, 3))
        # sum_one = np.sum(self.incoming_grad, axis=(0, 2, 3))
        # # print "SUM ONE:", sum_one.shape, sum_one
        # self.layer_grad = sum_one #np.sum(sum_one, axis=0)
        # # print "UPDATE GRAD:", self.layer_grad, self.layer_grad.shape

        layer_update = optimizer.getUpdate("convfilter", layer_grad)
