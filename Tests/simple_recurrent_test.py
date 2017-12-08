from npann.Utilities.misc_functions import *
from npann.Layers.Recurrent.SimpleRecurrent import SimpleRecurrent
from npann.Layers.Activations.Tanh import Tanh
from npann.Losses.MSE import MSE
from npann.Optimizers.SimpleGradientDescent import SimpleGradientDescent
from npann.Optimizers.RMSProp import RMSProp
import math

def sineSequence(start, end, step):
    seq = []
    val = start
    while val < end:
        seq.append([math.sin(val)])
        val += step
    # print seq
    return seq

if __name__ == "__main__":
    operand_size = 1
    hidden_size = 2
    sequence_length = 30
    number_epochs = 10000

    start = 0.0
    end = 3 * math.pi
    step = 0.01


    net = SimpleRecurrent(sequence_length, operand_size, hidden_size)
    opt = RMSProp(learning_rate=1E-2)
    seq = sineSequence(start, end, step)

    for epoch in range(number_epochs):
        r = random.randrange(0, len(seq) - sequence_length - 1)
        curr_seq = np.array(seq[r:r+sequence_length])
        answer = seq[r+sequence_length+1]
        output = net.forward(curr_seq)
        error = answer - output
        grad = net.backward(error)
        net.update(opt)
        print "EPOCH %d: %.8f" % (epoch, np.linalg.norm(error))

    print "TEST"
    r = random.randrange(0, len(seq) - sequence_length - 1)
    curr_seq = seq[r:r+sequence_length]
    answer = seq[r+sequence_length+1]

    val = start
    seq = []
    out = []
    vals = []
    while val < end:
        vals.append(val)
        seq.append([math.sin(val)])
        if len(vals) > sequence_length:
            o = net.forward(np.array(seq[-sequence_length:]))
            out.append(o)
        val += step

    import matplotlib.pyplot as plt
    plt.plot(vals, seq, 'r--', vals[-len(out):], out, 'g^')
    plt.show()
