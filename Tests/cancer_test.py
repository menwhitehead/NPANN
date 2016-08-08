from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Layers.Tanh import Tanh
from Layers.Relu import Relu
from Layers.LeakyRelu import LeakyRelu
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.AdaGrad import AdaGrad
from Optimizers.RMSProp import RMSProp


if __name__ == "__main__":
    lr = 0.01
    ann = Sequential()
    ann.addLayer(Dense(9, 25))
    ann.addLayer(Relu())
    ann.addLayer(Dense(25, 1))
    ann.addLayer(Relu())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancer()#Tanh() #loadXOR()    
    minibatch_size = 32
    number_epochs = 1000

    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
    print accuracyBinary(ann, X, y)
    # print ann.forward(X)
    # print y
