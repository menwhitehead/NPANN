from misc_functions import *
from Layers.Recurrent.LSTM import LSTM
from Layers.Activations.Tanh import Tanh
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.RMSProp import RMSProp

import struct

def floatToBits(f):
    s = struct.pack('>f', f)
    return bin(struct.unpack('>l', s)[0])

def sineSequence(start, end, step):
    seq = []
    val = start
    while val < end:
        seq.append(math.sin(val))
        val += step
    # print seq
    return seq

if __name__ == "__main__":
    operand_size = 1
    hidden_size = 1
    sequence_length = 30
    number_epochs = 10000

    start = 0.0
    end = 3.14
    step = 0.001


    net = LSTM(sequence_length, operand_size, hidden_size)
    act = Tanh()
    opt = RMSProp()
    seq = sineSequence(start, end, step)

    for epoch in range(number_epochs):
        r = random.randrange(0, len(seq) - sequence_length - 1)
        curr_seq = seq[r:r+sequence_length]
        # print "CURR SEQ:", curr_seq

        answer = seq[r+sequence_length+1]

        output = net.forward(curr_seq)
        act_output = act.forward(output)
        # print act_output, answer
        error = answer - act_output
         # error = np.repeat(error, 2)
        act_error = act.backward(error)
        grad = net.backward(act_error)
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
        seq.append(math.sin(val))
        if len(vals) > sequence_length:
            o = net.forward(seq[-sequence_length:])
            a = act.forward(o)
            out.append(a)
            # print a,
        val += step

    print vals

    import matplotlib.pyplot as plt

    # red dashes, blue squares and green triangles
    plt.plot(vals, seq, 'r--', vals[-len(out):], out, 'g^')
    plt.show()
