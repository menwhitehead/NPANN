from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Losses.MSE import MSE



if __name__ == "__main__":
    lr = 0.005
    ann = Sequential()
    ann.addLayer(Dense(9, 25, learning_rate=lr, activation='tanh', weight_init="glorot_normal"))
    drop = Dropout(25,25)
    ann.addLayer(drop)
    ann.addLayer(Dense(25, 1, learning_rate=lr, activation='tanh', weight_init="glorot_normal"))
    ann.addLoss(MSE())
    X, y = loadBreastCancerTanh() #loadXOR()
    
    perm = np.random.permutation(range(len(X)))
    mixedX = X[perm]
    mixedy = y[perm]
    
    train_percent = 0.95
    train_cutoff = int(train_percent * len(X))
    
    trainX, testX = mixedX[:train_cutoff], mixedX[train_cutoff:]
    trainy, testy = mixedy[:train_cutoff], mixedy[train_cutoff:]

    minibatch_size = 4
    number_epochs = 100

    for i in range(100):
        ann.train(trainX, trainy, minibatch_size, number_epochs, verbose=False)
        print ann.accuracyBinary(trainX, trainy), 
        drop.active = False
        print ann.accuracyBinary(testX, testy)
        drop.active = True
    # print ann.forward(X)
    # print y
