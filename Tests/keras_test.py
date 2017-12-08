import numpy as np
# np.random.seed(42)

import h5py
import random
from npann.Utilities.misc_functions import *

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


# def convertToOneHot(val, size):
#     x = np.zeros(size)
#     x[val] = 0.9
#     return x

def createModelXOR():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def createModelMNIST():
    model = Sequential()
    model.add(Dense(10, input_shape=(784,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":

    X, y = loadXOR() #loadMNIST()

    minibatch_size = 4

    model = createModelXOR()

    model.compile(#loss='categorical_crossentropy',
              loss='mse',
              #optimizer=RMSprop())
              optimizer=SGD(lr=0.9, momentum=0.0, nesterov=False, decay=0.0))
              #metrics=['accuracy'])

    # for layer in model.layers:
    #     weights = layer.get_weights()
    #     print weights
    #sys.exit(1)

    #micro = X[:1]
    #print model.predict(micro, batch_size=1, verbose=1)

    history = model.fit(X, y,
                batch_size=minibatch_size, nb_epoch=2500,
                verbose=1)

    print model.predict(X, batch_size=1, verbose=1)
