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

    def npconvolve(self, imgs, filters):
        outputs = []
        for img in imgs:
            for channel_number in range(len(img)):
                filter_outputs = []
                for filter_number in range(len(filters)):
                    conv_filter = filters[filter_number]
                    filter_outputs.append(convolve2d(img[channel_number], conv_filter, mode="same"))
                outputs.append(filter_outputs)
        outputs = np.array(outputs)
        return outputs

    def baseConvolve(self, arr1, arr2):
        return np.sum(arr1 * arr2)

    def convolve(self, image, filt):
        "Use filt as a convolution filter throughout image"
        output = np.zeros_like(image)
        image = np.pad(image, 1, 'constant', constant_values=(0,))
        for i in range(1, image.shape[2]-1):
            for j in range(1, image.shape[3]-1):
                tmp_arr = image[:, :, i-1:i+2, j-1:j+2]  # TODO: variable size kernels!
                convolution = self.baseConvolve(tmp_arr, filt)
                output[:, :, i-1, j-1] = convolution
        return output

    def convolveAll(self, arr1, arr2):
        output = np.zeros((arr1.shape[0], len(arr2), arr1.shape[2], arr1.shape[3]))
        for j in range(len(arr2)):
            output[j] = self.convolve(arr1, arr2[j])
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
        # print "FORWARD SHAPES:", self.incoming_acts.shape, self.outgoing_acts.shape
        return self.outgoing_acts

    def backward(self, incoming_grad):
        self.incoming_grad = incoming_grad
        # roted = np.copy(self.incoming_grad)
        #np.rot90(roted, 2)
        # print self.incoming_grad
        # np.rot90(self.incoming_grad, 2)
        # print self.incoming_grad
        self.outgoing_grad = self.npconvolve(self.incoming_grad, self.weights)
        # print "BACKWARD SHAPES:", self.incoming_grad.shape, self.outgoing_grad.shape
        return self.outgoing_grad

    def getLayerDerivatives(self):
        # return np.sum(self.incoming_acts * self.incoming_grad, axis=(0, 2, 3))
        # outs = self.convolveAll(self.incoming_acts, self.incoming_grad)
        # self.incoming_acts, self.incoming_grad, axis=(0, 2, 3))
        
        # self.incoming_grad = np.rot90(self.incoming_grad, 2)
        
        n_imgs = self.incoming_grad.shape[0]

        n_channels_convout = self.number_filters
        n_channels_imgs = self.incoming_acts.shape[1]
        # print n_channels_imgs
        fil_h = self.kernel_size
        fil_w = self.kernel_size
        fil_mid_h = fil_h / 2
        fil_mid_w = fil_w / 2
        
        filters_grad = np.zeros_like(self.weights)
        img_grad = np.zeros_like(self.outgoing_grad)

        padded = np.pad(np.copy(self.incoming_acts[0][0]), 1, mode="constant")
        print padded
        img_h = padded.shape[0]-2
        img_w = padded.shape[1]-2

        for y in range(1, img_h-1):
            for x in range(1, img_w-1):
                
                filters_grad[0, :, :] += self.incoming_acts[0, 0, y-1:y+2, x-1:x+2] * self.incoming_grad[0, 0, y, x]

                
                # for y_off in range(-1, 2):
                #     for x_off in range(-1, 2):
                #         img_y = y + y_off
                #         img_x = x + x_off
                #         fil_y = fil_mid_w + y_off
                #         fil_x = fil_mid_h + x_off
                #         print "PROCESSING: ", x, y, img_x, img_y, fil_x, fil_y
                #         filters_grad[0, fil_y, fil_x] += self.incoming_acts[0, 0, img_y, img_x] * self.incoming_grad[0, 0, y, x]
        #filters_grad /= n_imgs

        print filters_grad
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
