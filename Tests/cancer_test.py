from npann.Utilities.misc_functions import *
from npann.Models.Sequential import Sequential
from npann.Layers.Dense import Dense
from npann.Layers.Dropout import Dropout
from npann.Layers.Activations.Tanh import Tanh
from npann.Layers.Activations.Sigmoid import Sigmoid
from npann.Layers.Activations.Relu import Relu
from npann.Losses.MSE import MSE
from npann.Optimizers.SimpleGradientDescent import SimpleGradientDescent
from npann.Optimizers.AdaGrad import AdaGrad
from npann.Optimizers.RMSProp import RMSProp

if __name__ == "__main__":
    lr = 0.01
    ann = Sequential()
    ann.addLayer(Dense(9, 25))
    ann.addLayer(Relu())
    ann.addLayer(Dense(25, 25))
    ann.addLayer(Relu())
    ann.addLayer(Dense(25, 1))
    ann.addLayer(Relu())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancer()#Tanh() #loadXOR()
    minibatch_size = 100
    number_epochs = 10000

    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
    print accuracyBinary(ann, X, y)
