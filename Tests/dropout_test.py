from npann.Utilities.misc_functions import *
from npann.Models.Sequential import Sequential
from npann.Layers.Dense import Dense
from npann.Layers.Dropout import Dropout
from npann.Losses.MSE import MSE
from npann.Layers.Activations.Relu import Relu
from npann.Optimizers.RMSProp import RMSProp

if __name__ == "__main__":
    lr = 0.005
    ann = Sequential()
    ann.addLayer(Dense(9, 25))
    ann.addLayer(Relu())
    drop = Dropout(25,25, 0.5)
    ann.addLayer(drop)
    ann.addLayer(Dense(25, 1))
    ann.addLayer(Relu())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp(learning_rate = lr))
    X, y = loadBreastCancer() #loadXOR()

    perm = np.random.permutation(range(len(X)))
    mixedX = X[perm]
    mixedy = y[perm]

    train_percent = 0.90
    train_cutoff = int(train_percent * len(X))

    trainX, testX = mixedX[:train_cutoff], mixedX[train_cutoff:]
    trainy, testy = mixedy[:train_cutoff], mixedy[train_cutoff:]

    minibatch_size = 32
    number_epochs = 1000

    for i in range(100):
        ann.train(trainX, trainy, minibatch_size, number_epochs, verbose=0)
        train_accuracy = accuracyBinary(ann, trainX, trainy, train=True)
        test_accuracy = accuracyBinary(ann, testX, testy, train=False)
        print "Train acc: %6.2f%%, Test acc: %6.2f%%" % (train_accuracy, test_accuracy)
