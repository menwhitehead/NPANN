from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.AdaGrad import AdaGrad
from Optimizers.RMSProp import RMSProp



if __name__ == "__main__":
    lr = 0.01
    ann = Sequential()
    ann.addLayer(Dense(9, 15, activation='tanh', weight_init="glorot_normal"))
    # ann.addLayer(Dropout(5, 5))
    ann.addLayer(Dense(15, 1, activation='tanh', weight_init="glorot_normal"))
    ann.addLoss(MSE())
    #opt = SimpleGradientDescent(learning_rate = lr, momentum = 0.9)
    #opt = AdaGrad(learning_rate = 0.1)
    ann.addOptimizer(RMSProp(learning_rate = lr))

    X, y = loadBreastCancerTanh() #loadXOR()    
    minibatch_size = 16
    number_epochs = 10000

    ann.train(X, y, minibatch_size, number_epochs, verbose=2)
    print accuracyBinary(ann, X, y)
    # print ann.forward(X)
    # print y
