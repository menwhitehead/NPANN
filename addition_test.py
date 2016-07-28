from misc_functions import *
from Models.SoftBBFN import SoftBBFN
from Models.Sequential import Sequential
from Layers.Dense import RecurrentDense
from Layers.AiboPG import *
from Losses.MSE import MSE


if __name__ == "__main__":
    lr = 0.01 
    operand_size = 8
    hidden_size = 24
    
    dataset_size = 1000
    minibatch_size = 1
    epochs = 10000
   
    input_size = operand_size * 2
    output_size = operand_size * 2    

    hidden = RecurrentDense(input_size, hidden_size, learning_rate=lr, activation='sigmoid')
    output = RecurrentDense(hidden_size, output_size, learning_rate=lr, activation='sigmoid')
    model = Sequential()
    model.addLayer(hidden)
    model.addLayer(output)
    model.addLoss(MSE())

    X, y = loadAddition(dataset_size, operand_size)
    model.train(X, y, minibatch_size, epochs, verbose=True)
    # output = model.forward(X)
    # for i in range(len(output)):
    #     print output[i], y[i]





