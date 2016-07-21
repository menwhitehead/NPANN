from misc_functions import *
from Layers.Dense import Dense
from Layers.FiniteDifference import FiniteDifference
from Models.Sequential import Sequential
from Losses.MSE import MSE


if __name__ == "__main__":
    ann = Sequential()
    ann.addLayer(Dense(9, 30, activation='sigmoid', weight_init="glorot_normal"))
    #ann.addLayer(Dense(30, 1, activation='sigmoid', weight_init="glorot_normal"))
    #ann.addLayer(Reinforce(9, 3, activation='sigmoid', weight_init="glorot_normal"))
    ann.addLayer(FiniteDifference(30, 1, activation='sigmoid', weight_init="glorot_normal"))
    
    ann.addLoss(MSE())
    # ann.addOptimizer('sgd')
    # sys.exit(1)
    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 4
    epochs_per_chunk = 1000
    number_epochs = 50000

    for epoch in range(number_epochs / epochs_per_chunk):
        # ann.layers[0].resetTables()
        #ann.layers[1].resetTables()
        
        ann.train(X, y, minibatch_size, epochs_per_chunk, verbose=False)
        output = ann.forward(X)

        corr = 0
        for i in range(len(output)):
            if (output[i] < 0.5 and y[i] == 0) or (output[i] >= 0.5 and y[i] == 1):
                corr += 1
        print "ACCURACY:", corr / float(len(output))
