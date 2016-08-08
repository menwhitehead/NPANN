from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Losses.MSE import MSE

np.random.seed(42)

if __name__ == "__main__":
    lr = 0.005
    ann = Sequential()
    ann.addLayer(Dense(3, 1))
    ann.addLoss(MSE())
    X, y = loadXOR()    
    minibatch_size = 1
    number_epochs = 1
    
    X = X[:minibatch_size]
    y = y[:minibatch_size]
    
    epsilon = 0.0001
    
    output = ann.forward(X)
    ann.backward(output, y)
    layer = ann.layers[0]
    ann.loss_layer.calculateGrad(output, y)
    ann.update()
    
    len_weights1, len_weights2 = layer.weights.shape
    numeric_grads = np.zeros(layer.weights.shape)
    
    orig_weights = layer.weights

    for w1 in range(len_weights1): 
        for w2 in range(len_weights2):

            print w1, w2
            layer.weights[w1][w2] += epsilon
            output_pos = ann.forward(X)
            print "POS OUT:", output_pos
            pos_loss = ann.loss_layer.calculateLoss(output_pos, y)
            
            layer.weights[w1][w2] -= 2 * epsilon
            output_neg = ann.forward(X)
            print "NEG OUT:", output_neg
            neg_loss = ann.loss_layer.calculateLoss(output_neg, y)
            print output_pos, output_neg, pos_loss, neg_loss, y
            
            numeric_grads[w1][w2] = (pos_loss - neg_loss) / (2.0 * epsilon)

            #print "%7.4f %7.4f" % (numeric_grad[0][0], layer0_grad[w1][w2])
            layer.weights[w1][w2] += epsilon

print
print
for i in range(len(numeric_grads)):
    print "%7.4f %7.4f" % (numeric_grads[i][0], layer0_grad[i][0])








