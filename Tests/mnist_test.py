from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from Optimizers.RMSProp import RMSProp


if __name__ == "__main__":
    lr = 0.001
    ann = Sequential()
    ann.addLayer(Dense(784, 150))
    ann.addLayer(Dense(150, 10))
    
    # ann.addLayer(Dense(784, 256))
    # ann.addLayer(Dense(256, 256))
    # ann.addLayer(Dense(256, 10))
    
    # TEST!
    
    ann.addLoss(CategoricalCrossEntropy())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    
    X, y = loadMNIST()
    dataset_size = len(X)
    
    number_epochs = 1000
    accuracy_report_freq = 5
    minibatch_size = 128

    for i in range(number_epochs / accuracy_report_freq):
        ann.train(X, y, minibatch_size, accuracy_report_freq)
        print accuracy(ann, X, y)

            

    
    
    
