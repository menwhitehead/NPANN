from npann.Utilities.misc_functions import *
from npann.Layers.Dense import Dense
from npann.Layers.Reinforce import Reinforce
from npann.Layers.Activations.Relu import Relu
from npann.Layers.Activations.Tanh import Tanh
from npann.Models.SequentialReinforcement import SequentialReinforcement
from npann.Optimizers.RMSProp import RMSProp
from npann.Losses.MSE import MSE

def getReward(ann, X, y):
    output = ann.forward(X)
    loss = ann.loss_layer.calculateLoss(output, y)
    reward = 1 - (np.abs(loss)/np.max(loss))
    return reward

if __name__ == "__main__":
    ann = SequentialReinforcement()
    ann.addLayer(Dense(9, 20))
    ann.addLayer(Tanh())
    ann.addLayer(Dense(20, 1))
    ann.addLayer(Tanh())
    ann.addLayer(Reinforce(1, std_dev=1.51))
    ann.addLayer(Tanh())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp())

    X, y = loadBreastCancerTanh()
    minibatch_size = 32
    number_epochs = 100000

    ann.train(X, y, minibatch_size, number_epochs, getReward, verbose=2)
