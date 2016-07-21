from npann_functions import *
from Models.Graph import Graph
from Layers.Dense import Dense

if __name__ == "__main__":
    ann = Graph()
    ann.addLayer("dense1", Dense(9, 30), ["input1"])
    ann.addLayer("dense2", Dense(30, 1), ["dense1"], is_output=True)
    ann.addLoss(MSE())

    X, y = loadBreastCancer() #loadXOR()
    minibatch_size = 4
    epochs_per_chunk = 1000
    number_epochs = 50000

    for epoch in range(number_epochs / epochs_per_chunk):
        ann.train(X, y, minibatch_size, epochs_per_chunk, verbose=False)
        output = ann.forward(X)

        corr = 0
        for i in range(len(output)):
            if (output[i] < 0.5 and y[i] == 0) or (output[i] >= 0.5 and y[i] == 1):
                corr += 1
        print "ACCURACY:", corr / float(len(output))
