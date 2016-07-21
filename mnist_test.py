from misc_functions import *
from Layers.Dense import Dense
from Models.Sequential import Sequential
from Losses.CategoricalCrossEntropy import CategoricalCrossEntropy


if __name__ == "__main__":
    
    ann = Sequential()
    ann.addLayer(Dense(784, 10))
    ann.addLayer(Dense(10, 10))
    
    # ann.addLayer(Dense(784, 256))
    # ann.addLayer(Dense(256, 256))
    # ann.addLayer(Dense(256, 10))
    
    ann.addLoss(CategoricalCrossEntropy())
    
    X, y = loadMNIST()
    dataset_size = len(X)
    
    number_epochs = 1000
    accuracy_report_freq = 5
    minibatch_size = 10

    for i in range(number_epochs / accuracy_report_freq):
        ann.train(X, y, minibatch_size, accuracy_report_freq)
        ann.accuracy(X, y)

            

    
    
    
