from misc_functions import *
from Models.Sequential import Sequential
from Layers.Dense import Dense
from Layers.Dropout import Dropout
from Layers.Activations.Tanh import Tanh
from Optimizers.RMSProp import RMSProp
from Losses.MSE import MSE

np.random.seed(42)

if __name__ == "__main__":
    lr = 0.005
    ann = Sequential()
    ann.addLayer(Dense(9, 3))
    ann.addLayer(Tanh())
    ann.addLayer(Dense(3, 1))
    ann.addLayer(Tanh())
    ann.addLoss(MSE())
    ann.addOptimizer(RMSProp())
    
    X, y = loadBreastCancerTanh() #loadXOR()    
    minibatch_size = 1
    number_epochs = 1
    
    X = X[:minibatch_size]
    y = y[:minibatch_size]
    
    epsilon = 0.0001
    
    output = ann.forward(X)
    ann.backward(output, y)
    ann.loss_layer.calculateGrad(output, y)
    ann.update()
    print "LAYER GRAD:", ann.layers[0].layer_grad
    print "LAYER GRAD:", ann.layers[2].layer_grad


    numeric_grads = []
        
    for layer in range(len(ann.layers)):
        print
        # print "Layer:", layer
        # print "Weights:", ann.layers[layer].weights.shape
        for w in range(len(ann.layers[layer].weights)): # -1 for bias
            # Numeric check
            #print ann.layers[0].weights
            wcopy_pos = np.copy(ann.layers[layer].weights)
            wcopy_neg = np.copy(ann.layers[layer].weights)
                
            for w2 in range(len(wcopy_pos[0])): # -1 for bias   

                #print "POS:", wcopy_pos
                # print w, w2
                wcopy_pos[w][w2] += epsilon
                ann.layers[layer].weights = wcopy_pos
                output_pos = ann.forward(X)
                pos_loss = ann.loss_layer.calculateLoss(output_pos, y)
                
                wcopy_neg[w][w2] -= epsilon
                ann.layers[layer].weights = wcopy_neg
                output_neg = ann.forward(X)
                neg_loss = ann.loss_layer.calculateLoss(output_neg, y)
                
                numeric_grad = (pos_loss - neg_loss) / (2.0 * epsilon)
                #numeric_grads.append(numeric_grad)
                #print "%7.4f %7.4f" % (numeric_grad[0][0], backprop_grads[layer][0][w]) #grad[0][w]
                print numeric_grad[0][0]
                
        # print "Backprop: ", grad
        #print "Numeric:  ", numeric_grads
    
    
    
    
    
    
    
    
        
        
        