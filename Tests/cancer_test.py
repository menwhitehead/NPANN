from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Losses.MSE import MSE



if __name__ == "__main__":
    lr = 0.005
    ann = Sequential()
    ann.addLayer(Dense(9, 15, learning_rate=lr, activation='tanh', weight_init="glorot_normal"))
    # ann.addLayer(Dropout(5, 5))
    ann.addLayer(Dense(15, 1, learning_rate=lr, activation='tanh', weight_init="glorot_normal"))
    ann.addLoss(MSE())
    X, y = loadBreastCancerTanh() #loadXOR()    
    minibatch_size = 1
    number_epochs = 100

    for i in range(100):
        ann.train(X, y, minibatch_size, number_epochs, verbose=False)
        print ann.accuracyBinary(X, y)
    # print ann.forward(X)
    # print y
