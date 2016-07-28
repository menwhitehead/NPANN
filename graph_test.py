from misc_functions import *
from Models.Graph import Graph
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Merge import Merge

from Losses.MSE import MSE

def graphTest():
    lr = 0.1
    ann = Graph()
    ann.addLayer("dense1", Dense(9, 15, learning_rate=lr), ["input1"])
    ann.addLayer("dense2a", Dense(15, 3, learning_rate=lr), ["dense1"])
    ann.addLayer("dense2b", Dense(15, 3, learning_rate=lr), ["dense1"])
    ann.addLayer("merger", Merge(), ["dense2a", "dense2b"])
    ann.addLayer("dense3", Dense(3, 1, learning_rate=lr), ["merger"], is_output=True)
    ann.addLoss(MSE())

    X, y = loadBreastCancer()
    minibatch_size = 4
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=True)

def comparisonSequentialTest():
    lr = 0.1
    ann = Sequential()
    ann.addLayer(Dense(9, 15, learning_rate=lr))
    ann.addLayer(Dense(15, 1, learning_rate=lr))
    ann.addLoss(MSE())

    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 4
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=True)
    #ann.accuracy(X, y)


if __name__ == "__main__":
    # comparisonSequentialTest()
    graphTest()

