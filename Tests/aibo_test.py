from npann.Utilities.misc_functions import *
from npann.Layers.Dense import Dense
from npann.Layers.AiboPG import *
from npann.Layers.Activations.Sigmoid import Sigmoid
from npann.Models.Sequential import Sequential
from npann.Optimizers.RMSProp import RMSProp
from npann.Losses.MSE import MSE

if __name__ == "__main__":
    lr = 0.01
    ann = Sequential()
    ann.addLayer(Dense(9, 30))
    ann.addLayer(Sigmoid())
    ann.addLayer(AiboPG2(30, 1))
    ann.addLayer(Sigmoid())
    ann.addOptimizer(RMSProp(learning_rate = lr))
    ann.addLoss(MSE())

    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 10
    number_epochs = 10000

    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
