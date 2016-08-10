from misc_functions import *
from Models.SoftBBFN import SoftBBFN
from Layers.RecurrentDense import RecurrentDense
from Layers.Activations.Relu import Relu
from Layers.Activations.Sigmoid import Sigmoid
from Losses.MSE import MSE
from Optimizers.RMSProp import RMSProp

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
    # print list(x)
    z = binaryToInt(x) + 1

    # x = list(x).index(1.0)
    #y = list(y).index(0.9)
    # z = x + 1
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
    lr = 0.01
    operand_size = 8
    applying_output_size = 8
    hidden_size = 16

    dataset_size = 1000
    minibatch_size = 1
    epochs = 10000

    # function_library = [addThem, randomResult, randomResult,randomResult, randomResult,randomResult, randomResult,randomResult, randomResult]
    function_library = [addThem, addOne, randomResult]

    applying_input_size = len(function_library) * operand_size * 2
    #applying_output_size = operand_size * 2
    hidden_input_size = hidden_size + applying_output_size
    output_input_size = hidden_size

    print "SIZES:", applying_input_size, hidden_input_size, output_input_size

    applying = RecurrentDense(applying_input_size, applying_output_size)
    applying_act = Sigmoid()
    hidden = RecurrentDense(hidden_input_size, hidden_size)
    hidden_act = Sigmoid()
    output = RecurrentDense(hidden_size, 2*operand_size)
    output_act = Sigmoid()

    model = SoftBBFN(applying, applying_act, hidden, hidden_act, output, output_act, function_library, sequence_length=2)
    model.addOptimizer(RMSProp())
    model.addLoss(MSE())

    X, y = loadAdditionPlusOneBinary(dataset_size, operand_size)
    model.train(X, y, minibatch_size, epochs, verbose=2)
