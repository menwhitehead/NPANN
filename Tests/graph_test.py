from misc_functions import *
from Models.Graph import Graph
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Merge import Merge
from Layers.Dropout import Dropout
from Layers.Activations.Tanh import Tanh

from Losses.MSE import MSE
from Optimizers.RMSProp import RMSProp

#np.random.seed(42)

def graphTest():
    ann = Graph()
    ann.addLayer("dense1", Dense(9, 22), ["input1"])
    ann.addLayer("act1", Tanh(), ["dense1"])
    
    ann.addLayer("dense2a", Dense(22, 12), ["act1"])
    ann.addLayer("act2a", Tanh(), ["dense2a"])

    ann.addLayer("dense2b", Dense(22, 12), ["act1"])
    ann.addLayer("act2b", Tanh(), ["dense2b"])

    ann.addLayer("merger", Merge(24), ["act2a", "act2b"])
    # ann.addLayer("merger", Merge(), ["dense1"])
    # ann.addLayer("dense3", Dense(3, 1), ["merger"], is_output=True)
    
    ann.addLayer("dense3", Dense(24, 1), ["merger"])
    ann.addLayer("act3", Tanh(), ["dense3"], is_output=True)
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp())

    X, y = loadBreastCancerTanh()
    minibatch_size = 16
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
    print "Final accuracy: ", accuracyBinary(ann, X, y)

def comparisonSequentialTest():
    ann = Sequential()
    ann.addLayer(Dense(9, 25))
    ann.addLayer(Tanh())
    ann.addLayer(Dense(25, 1))
    ann.addLayer(Tanh())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp())

    X, y = loadBreastCancerTanh() #loadXOR()
    minibatch_size = 16
    number_epochs = 1000
    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
    #print ann.accuracyBinary(X, y)
    print accuracyBinary(ann, X, y)


if __name__ == "__main__":
    # comparisonSequentialTest()
    graphTest()
    # 
