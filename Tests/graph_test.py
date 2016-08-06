from misc_functions import *
from Models.Graph import Graph
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Merge import Merge
from Layers.Dropout import Dropout
from Losses.MSE import MSE
from Optimizers.RMSProp import RMSProp

#np.random.seed(42)

def graphTest():
    lr = 0.01
    ann = Graph()
    ann.addLayer("dense1", Dense(9, 22, activation="tanh"), ["input1"])
    ann.addLayer("dense2a", Dense(22, 12), ["dense1"])
    ann.addLayer("dense2b", Dense(22, 12), ["dense1"])
    ann.addLayer("merger", Merge(24), ["dense2a", "dense2b"])
    # ann.addLayer("merger", Merge(), ["dense1"])
    # ann.addLayer("dense3", Dense(3, 1), ["merger"], is_output=True)
    ann.addLayer("dense3", Dense(24, 1, activation="tanh"), ["merger"], is_output=True)
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancerTanh()
    minibatch_size = 4
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=True)

def comparisonSequentialTest():
    lr = 0.01
    ann = Sequential()
    ann.addLayer(Dense(9, 25, activation="tanh"))
    ann.addLayer(Dense(25, 1, activation="tanh"))
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 4
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=True)
    print ann.accuracyBinary(X, y)


if __name__ == "__main__":
    # comparisonSequentialTest()
    graphTest()
    # 
