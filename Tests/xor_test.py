from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Activations.Sigmoid import Sigmoid
from Layers.Activations.Tanh import Tanh
from Layers.Activations.Relu import Relu
from Losses.MSE import MSE
from Optimizers.RMSProp import RMSProp


if __name__ == "__main__":
    ann = Sequential()
    ann.addLayer(Dense(3, 5))
    ann.addLayer(Relu())
    ann.addLayer(Dense(5, 1))
    ann.addLayer(Relu())    
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp())

    X, y = loadXOR()    
    minibatch_size = 1
    number_epochs = 10000

    ann.train(X, y, minibatch_size, number_epochs, verbose=1)
