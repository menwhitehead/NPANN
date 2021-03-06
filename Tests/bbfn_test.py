from misc_functions import *
from Models.BBFN import BBFN
from Models.SequentialReinforcement import SequentialReinforcement
from Layers.RecurrentDense import RecurrentDense
from Layers.ReinforceRecurrent import ReinforceRecurrent
from Layers.Dense import Dense
from Layers.Activations.Relu import Relu
from Layers.Activations.Tanh import Tanh
from Layers.Activations.Softmax import Softmax
from Layers.Activations.Sigmoid import Sigmoid
from Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
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
    x = list(x).index(1)
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
    operand_size = 8
    hidden_size = 4*operand_size
    dataset_size = 1000
    minibatch_size = 256
    epochs = 10000

    # function_library = [addThem, addOne, addOne, addOne, addOne, addOne, addOne]
    function_library = [addThem, randomResult, randomResult, randomResult]
    #function_library = [addThem, randomResult]

    applying = RecurrentDense(2*operand_size + len(function_library), 2*operand_size)
    applying_act = Sigmoid()
    hidden = RecurrentDense(hidden_size + 2*operand_size, hidden_size)
    hidden_act = Sigmoid()
    output = RecurrentDense(hidden_size, 2*operand_size)
    output_act = Sigmoid()
    opt = RMSProp()

    # Network for choosing the next function to call
    func_network = SequentialReinforcement()
    func_network.addLayer(RecurrentDense(hidden_size, len(function_library)))
    func_network.addLayer(Sigmoid())
    func_network.addLayer(ReinforceRecurrent(len(function_library), std_dev=0.11))
    func_network.addLayer(Softmax())
    func_network.addOptimizer(opt)

    # Network for choosing the part of the expression to pay attention to
    exp_network = SequentialReinforcement()
    exp_network.addLayer(RecurrentDense(hidden_size, len(function_library)))
    func_network.addLayer(Relu())
    exp_network.addLayer(ReinforceRecurrent(len(function_library), std_dev=0.11))
    exp_network.addLayer(Softmax())
    exp_network.addOptimizer(opt)


    model = BBFN(applying, applying_act, hidden, hidden_act,
                output, output_act, func_network, exp_network,
                function_library, sequence_length=2)

    model.addOptimizer(opt)
    # model.addLoss(CategoricalCrossEntropy())
    model.addLoss(MSE())

    X, y = loadAdditionBinary(dataset_size, operand_size)
    #print X, y
    model.train(X, y, minibatch_size, epochs, verbose=1)
    # output = model.forward(X)
    # for i in range(len(output)):
    #     print output[i], y[i]
