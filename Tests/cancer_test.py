from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Layers.Activations.Tanh import Tanh
from Layers.Activations.Sigmoid import Sigmoid
from Layers.Activations.Relu import Relu
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.AdaGrad import AdaGrad
from Optimizers.RMSProp import RMSProp

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
