from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Convolution import Convolution
from Layers.Flatten import Flatten
from Layers.Activations.Sigmoid import Sigmoid
from Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from Optimizers.RMSProp import RMSProp


if __name__ == "__main__":
    lr = 0.01
    image_width = 28
    image_height = 28
    number_filters = 1
    number_classes = 10

    ann = Sequential()
    ann.addLayer(Convolution(image_width, image_height, number_filters))
    ann.addLayer(Sigmoid())
    ann.addLayer(Flatten())
    ann.addLayer(Dense(number_filters * image_width * image_height, number_classes))
    ann.addLayer(Sigmoid())
    ann.addLoss(CategoricalCrossEntropy())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadMNIST2D()
    dataset_size = len(X)
    # print X[0]

    number_epochs = 1000
    accuracy_report_freq = 10
    minibatch_size = 16

    for i in range(number_epochs / accuracy_report_freq):
        ann.train(X, y, minibatch_size, accuracy_report_freq, verbose=1)
        print accuracy(ann, X, y)
