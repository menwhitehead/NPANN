from misc_functions import *
from Models.SoftBBFN import SoftBBFN
from Layers.Dense import RecurrentDense
from Layers.AiboPG import *
from Losses.MSE import MSE



# get an array with two 1-hot operands (representing integers) packed in
# add 'em and return a 1-hot result
def addThem(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    z = binaryToInt(x) + binaryToInt(y)
    #z = convertToOneHot(z, len(packed_operands))
    z = convertToBinary(z, len(packed_operands))
    return z

# get an array with two 1-hot operands (representing integers) packed in
# and add 1 to the first operand
def addOne(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    x = list(x).index(0.9)
    #y = list(y).index(0.9)
    z = x + 1
    # z = convertToOneHot(z, len(packed_operands))
    z = convertToBinary(z, len(packed_operands))
    return z

# get an array with two 1-hot operands (representing integers) packed in
# and add 1 to the first operand
def randomResult(packed_operands):
    x, y = packed_operands[:len(packed_operands)/2], packed_operands[len(packed_operands)/2:]
    z = random.randrange(len(packed_operands))
    # z = convertToOneHot(z, len(packed_operands))
    z = convertToBinary(z, len(packed_operands))
    return z


if __name__ == "__main__":
    lr = 0.05   # 0.05 worked instantly!
    operand_size = 8

    applying_size = 10
    hidden_size = 9
    
    dataset_size = 2
    minibatch_size = 2
    epochs = 10000

    # function_library = [addThem, addOne, addOne, addOne, addOne, addOne, addOne]
    function_library = [addThem, randomResult, randomResult, randomResult]
    
    applying_input_size = len(function_library) * (operand_size + 1) * 2
    hidden_input_size = hidden_size + len(function_library) * (operand_size) * 2

    applying = RecurrentDense(applying_input_size, applying_size, learning_rate=lr)

    hidden = RecurrentDense(hidden_input_size, hidden_size, learning_rate=lr)
    output = RecurrentDense(hidden_size, 2*operand_size, learning_rate=lr, activation='sigmoid')
    model = SoftBBFN(hidden, output, function_library, sequence_length=2)
    model.addLoss(MSE())

    X, y = loadAddition(dataset_size, operand_size)
    #print X, y
    model.train(X, y, minibatch_size, epochs, verbose=True)
    # output = model.forward(X)
    # for i in range(len(output)):
    #     print output[i], y[i]




