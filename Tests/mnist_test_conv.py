from npann.Utilities.misc_functions import *
from npann.Models.Sequential import Sequential
from npann.Layers.Dense import Dense
from npann.Layers.Convolution import Convolution
from npann.Layers.Flatten import Flatten
from npann.Layers.Activations.Sigmoid import Sigmoid
from npann.Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from npann.Optimizers.RMSProp import RMSProp

if __name__ == "__main__":
    lr = 1E-4
    image_width = 28
    image_height = 28
    number_filters = 10
    number_classes = 10
    number_epochs = 1000
    accuracy_report_freq = 3
    minibatch_size = 100

    ann = Sequential()
    ann.addLayer(Convolution(number_filters, image_width, image_height))
    ann.addLayer(Sigmoid())
    ann.addLayer(Flatten())
    ann.addLayer(Dense(number_filters * image_width * image_height, number_classes))
    ann.addLayer(Sigmoid())
    ann.addLoss(CategoricalCrossEntropy())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadMNIST2D()
    dataset_size = len(X)

    for i in range(number_epochs / accuracy_report_freq):
        ann.train(X, y, minibatch_size, accuracy_report_freq, verbose=1)
        print accuracy(ann, X, y)
