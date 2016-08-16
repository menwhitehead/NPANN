from misc_functions import *
from Layers.Recurrent.SimpleRecurrent2 import SimpleRecurrent2
from Layers.Dense import Dense

from Layers.Activations.Tanh import Tanh
from Layers.Activations.Softmax import Softmax
from Layers.Activations.Sigmoid import Sigmoid

from Losses.MSE import MSE
from Optimizers.SimpleGradientDescent import SimpleGradientDescent
from Optimizers.RMSProp import RMSProp
import string

ALPHABET_LENGTH = 27

def getVector(char):
    v = np.zeros(ALPHABET_LENGTH)
    if char.isalpha():
        v[ord(char)-97] = 1.0
    else:
        v[-1] = 1.0

    return v

def vectorToChar(v):
    # print v
    ind = v.argmax()
    # print ind
    if ind != ALPHABET_LENGTH - 1:
        return chr(ind + 97)
    else:
        return ' '

def getChunkVectors(text):
    vecs = []
    for c in text:
        vecs.append(getVector(c))
    return np.array(vecs)


def getSeq(txt, length=10):
    # Two sequences
    start = random.randrange(len(txt) - length - 1)
    sqs = getChunkVectors(txt[start:start+length])
    return sqs

if __name__ == "__main__":
    input_size = ALPHABET_LENGTH
    hidden_size = 256
    output_size = ALPHABET_LENGTH

    sequence_length = 50
    number_epochs = 100
    number_tests = 1000

    txt = open("wiki.txt", 'r').read()

    net = SimpleRecurrent2(sequence_length, input_size, hidden_size, output_size, backprop_limit=10)
    act = Softmax()
    opt = RMSProp(learning_rate=1E-4)

    for test in range(number_tests):
        errors = []
        for epoch in range(number_epochs):
            seq = getSeq(txt, sequence_length + 1)
            answer = seq[sequence_length]
            seq = seq[:sequence_length]

            # print seq
            # print seq.shape

            output = net.forward(seq)
            act_output = act.forward(output)

            # LOSS
            error = answer - act_output

            act_error = act.backward(error)
            grad = net.backward(act_error)
            net.update(opt)

            error = np.linalg.norm(error)
            errors.append(error)
            errors = errors[-100:]

            # print "EPOCH %d:%12.8f%12.8f" % (epoch, error, sum(errors)/len(errors))

        print "TEST %d" % test
        number_to_generate = 1000
        curr_seq = getSeq(txt, sequence_length)
        output = ""
        for seq in curr_seq:
            output += vectorToChar(seq)
        output += " -> "
        for i in range(number_to_generate):
                o = net.forward(curr_seq)
                a = act.forward(o)
                c = vectorToChar(a)
                output += c
                curr_seq = np.vstack((curr_seq[1:], getVector(c)))

        print output
        print