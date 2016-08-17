from misc_functions import *
from Layers.Recurrent.GRU import GRU
from Layers.Activations.Tanh import Tanh
from Layers.Activations.Relu import Relu
from Layers.Activations.Sigmoid import Sigmoid
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.RMSProp import RMSProp

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
    input_size = 1
    hidden_size = 2
    output_size = 1
    sequence_length = 50
    number_epochs = 500

    start = 0.0
    end = 3 * math.pi #3.14
    step = 0.01

    net = GRU(sequence_length, input_size, hidden_size, backprop_limit=5)
    act = Tanh()
    opt = RMSProp(learning_rate=1E-2)
    seq = sineSequence(start, end, step)
    errors = []

    for epoch in range(number_epochs):
        r = random.randrange(0, len(seq) - sequence_length - 1)
        curr_seq = np.array(seq[r:r+sequence_length])
        # print "CURR SEQ:", curr_seq

        answer = seq[r+sequence_length+1]
        output = net.forward(curr_seq)
        act_output = act.forward(output)
        # print curr_seq, answer, act_output

        error = answer - act_output
        act_error = act.backward(error)
        grad = net.backward(act_error)
        net.update(opt)

        error = np.linalg.norm(error)
        errors.append(error)
        errors = errors[-100:]

        print "EPOCH %d:%12.8f%12.8f" % (epoch, error, sum(errors)/len(errors))

    print "TEST"
    # r = random.randrange(0, len(seq) - sequence_length - 1)
    # curr_seq = np.array(seq[r:r+sequence_length])
    # answer = seq[r+sequence_length+1]

    val = start
    seq = []
    out = []
    vals = []
    while val < end:
        vals.append(val)
        seq.append([math.sin(val)])
        if len(vals) > sequence_length:
            curr = np.array(seq[-sequence_length:])
            o = net.forward(curr)
            a = act.forward(o)
            # print curr, a
            out.append(a)
            # print a,
        val += step

    import matplotlib.pyplot as plt
    plt.plot(vals, seq, 'r--', vals[-len(out):], out, 'g^')
    plt.show()
