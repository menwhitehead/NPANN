from misc_functions import *
from Layers.Dense import Dense
from Layers.FiniteDifference import FiniteDifference
from Layers.Activations.Sigmoid import Sigmoid
from Models.Sequential import Sequential
from Optimizers.RMSProp import RMSProp
from Losses.MSE import MSE


if __name__ == "__main__":
    ann = Sequential()
    ann.addLayer(Dense(9, 30))
    ann.addLayer(Sigmoid())
    ann.addLayer(FiniteDifference(30, 1))
    ann.addLayer(Sigmoid())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp())
    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 4
    number_epochs = 10000

    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
    