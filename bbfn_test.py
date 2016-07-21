from misc_functions import *
from Models.BBFN import BBFN
from Layers.Dense import Dense
from Layers.AiboPG import *
from Losses.CategoricalCrossEntropy import CategoricalCrossEntropy



# get an array with two 1-hot operands (representing integers) packed in
# add 'em and return a 1-hot result
def addThem(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    x = list(x).index(0.9)
    y = list(y).index(0.9)
    z = x + y
    z = convertToOneHot(z, len(packed_operands))
    return z

# get an array with two 1-hot operands (representing integers) packed in
# and add 1 to the first operand
def addOne(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    x = list(x).index(0.9)
    #y = list(y).index(0.9)
    z = x + 1
    z = convertToOneHot(z, len(packed_operands))
    return z

if __name__ == "__main__":
    
    applying = Dense(20, 20)
    hidden = Dense(50, 30)
    output = Dense(30, 20, activation='softmax')
    func = AiboPG2(30, 2, activation='none')
    exp = AiboPG2(30, 2, activation='none')
    function_library = [addThem] #[addThem, addOne]
    
    model = BBFN(applying, hidden, output, func, exp, function_library)
  
    # ann.addLoss(MSE())
    model.addLoss(CategoricalCrossEntropy())

    X, y = loadAddition(1000, 10)
    minibatch_size = 1
    epochs = 10000

    model.train(X, y, minibatch_size, epochs, verbose=True)
    # output = model.forward(X)
    # for i in range(len(output)):
    #     print output[i], y[i]

