from misc_functions import *
from Layers.Dense import Dense
from Layers.Reinforce import *
from Models.SequentialReinforcement import SequentialReinforcement
from Optimizers.RMSProp import RMSProp
from Losses.MSE import MSE

if __name__ == "__main__":
    ann = SequentialReinforcement()
    ann.addLayer(Dense(9, 20, activation='tanh'))
    ann.addLayer(Dense(20, 1, activation='tanh'))
    ann.addLayer(Reinforce(1, std_dev=0.11, activation='tanh'))

    # ann.addLayer(Dense(25, 1, activation='tanh'))
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate=0.01))

    X, y = loadBreastCancerTanh()
    minibatch_size = 10
    number_epochs = 100000

    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
