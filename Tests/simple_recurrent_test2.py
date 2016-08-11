from misc_functions import *
from Layers.Recurrent.SimpleRecurrent import SimpleRecurrent
from Layers.Activations.Tanh import Tanh
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
    operand_size = ALPHABET_LENGTH
    hidden_size = 10
    output_size = ALPHABET_LENGTH

    sequence_length = 20
    number_epochs = 10000

    txt = open("wiki.txt", 'r').read()

    net = SimpleRecurrent(sequence_length, operand_size, hidden_size)
    act = Tanh()
    net2 = Dense(hidden_size, output_size)
    act2 = Softmax()

    opt = RMSProp()

    for epoch in range(number_epochs):
        seq = getSeq(txt, sequence_length + 1)
        answer = seq[sequence_length]
        seq = seq[:sequence_length]
        output = net.forward(seq)
        act_output = act.forward(output)
        output2 = net2.forward(act_output)
        act_output2 = act2.forward(output2)
        # print act_output, answer

        # LOSS
        error = answer - act_output2

         # error = np.repeat(error, 2)
        #HEREHERE HEREHERE
        act_output2_error = act2.backward(error)

        act_error = act.backward(error)
        grad = net.backward(act_error)
        net.update(opt)

        print "EPOCH %d: %.8f" % (epoch, np.linalg.norm(error))

    print "TEST"
    number_to_generate = 100
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
