from npann.Utilities.misc_functions import *
from npann.Models.Sequential import Sequential
from npann.Layers.Dense import Dense
from npann.Layers.Activations.Sigmoid import Sigmoid
from npann.Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from npann.Optimizers.RMSProp import RMSProp


if __name__ == "__main__":
    lr = 0.01
    number_hidden = 10
    ann = Sequential()
    ann.addLayer(Dense(784, number_hidden))
    ann.addLayer(Sigmoid())
    ann.addLayer(Dense(number_hidden, 10))
    ann.addLayer(Sigmoid())
    ann.addLoss(CategoricalCrossEntropy())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadMNIST()
    dataset_size = len(X)

    number_epochs = 1000
    accuracy_report_freq = 5
    minibatch_size = 128

    for i in range(number_epochs / accuracy_report_freq):
        ann.train(X, y, minibatch_size, accuracy_report_freq, verbose=1)
        print accuracy(ann, X, y)
