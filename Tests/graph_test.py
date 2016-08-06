from misc_functions import *
from Models.Graph import Graph
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Merge import Merge
from Layers.Dropout import Dropout
from Losses.MSE import MSE
from Optimizers.RMSProp import RMSProp

def graphTest():
    lr = 0.01
    ann = Graph()
    ann.addLayer("dense1", Dense(9, 25), ["input1"])
    #ann.addLayer("dense2a", Dense(25, 3), ["dense1"])
    #ann.addLayer("dense2b", Dense(25, 3), ["dense1"])
    #ann.addLayer("merger", Merge(), ["dense2a", "dense2b"])
    #ann.addLayer("merger", Merge(), ["dense2a"])
    # ann.addLayer("dense3", Dense(3, 1), ["merger"], is_output=True)
    ann.addLayer("dense3", Dense(25, 1), ["dense1"], is_output=True)
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancer()
    minibatch_size = 4
    dataset_size = 1000
    number_epochs = 100000
    ann.train(X[:dataset_size], y[:dataset_size], minibatch_size, number_epochs, verbose=True)

def comparisonSequentialTest():
    lr = 0.1
    ann = Sequential()
    ann.addLayer(Dense(9, 15))
    ann.addLayer(Dense(15, 1))
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 4
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=True)
    #ann.accuracy(X, y)


if __name__ == "__main__":
    #comparisonSequentialTest()
    graphTest()
    # 
