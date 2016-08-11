from misc_functions import *
from Layers.Recurrent.SimpleRecurrent import SimpleRecurrent
from Layers.Activations.Tanh import Tanh
from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
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
    sequence_length = 10
    number_epochs = 10000

    net = SimpleRecurrent(sequence_length, operand_size, hidden_size)
    act = Tanh()
    opt = SimpleGradientDescent()
    seq = sineSequence(0, 30.14, 0.01)
    
    for epoch in range(number_epochs):
        r = random.randrange(0, len(seq) - sequence_length - 1)
        curr_seq = seq[r:r+sequence_length]
        answer = seq[r+sequence_length+1]
        
        output = net.forward(curr_seq)
        act_output = act.forward(output)
        error = answer - act_output
         # error = np.repeat(error, 2)
        act_error = act.backward(error)
        grad = net.backward(act_error)
        net.update(opt)
        
        print "EPOCH %d: %.8f" % (epoch, np.power(act_error, 2))
        
    print "TEST"
    r = random.randrange(0, len(seq) - sequence_length - 1)
    curr_seq = seq[r:r+sequence_length]
    answer = seq[r+sequence_length+1]
    
    val = 0
    seq = []
    out = []
    vals = []
    while val < 30.14:
        vals.append(val)
        seq.append(math.sin(val))
        if len(vals) > sequence_length:
            o = net.forward(vals[-sequence_length:])
            a = act.forward(o)
            out.append(a)
        val += 0.01
    
    
    import matplotlib.pyplot as plt

    # red dashes, blue squares and green triangles
    plt.plot(vals, seq, 'r--', vals[-len(out):], out, 'g^')
    plt.show()

    
    
    
