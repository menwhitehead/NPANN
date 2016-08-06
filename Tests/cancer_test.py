from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.AdaGrad import AdaGrad
from Optimizers.RMSProp import RMSProp



if __name__ == "__main__":
    lr = 0.005
    mom = 0.9
    ann = Sequential()
    ann.addLayer(Dense(9, 15, activation='tanh', weight_init="glorot_normal"))
    # ann.addLayer(Dropout(5, 5))
    ann.addLayer(Dense(15, 1, activation='tanh', weight_init="glorot_normal"))
    ann.addLoss(MSE())
    #opt = SimpleGradientDescent(learning_rate = lr, momentum = mom)
    # opt = AdaGrad(learning_rate = 0.1)
    opt = RMSProp(learning_rate = 0.01)

    ann.addOptimizer(opt)
    X, y = loadBreastCancerTanh() #loadXOR()    
    minibatch_size = 4
    number_epochs = 100

    for i in range(100):
        ann.train(X, y, minibatch_size, number_epochs, verbose=False)
        print ann.accuracyBinary(X, y)
    # print ann.forward(X)
    # print y
