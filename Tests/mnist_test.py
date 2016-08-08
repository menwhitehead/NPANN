from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Activations.Sigmoid import Sigmoid
from Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from Optimizers.RMSProp import RMSProp


if __name__ == "__main__":
    lr = 0.01
    ann = Sequential()
    ann.addLayer(Dense(784, 150))
    ann.addLayer(Sigmoid())
    ann.addLayer(Dense(150, 10))
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

            

    
    
    
