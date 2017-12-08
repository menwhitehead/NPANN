from npann.Utilities.misc_functions import *
from npann.Models.Sequential import Sequential
from npann.Layers.Dense import Dense
from npann.Layers.Activations.Sigmoid import Sigmoid
from npann.Losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from npann.Losses.MSE import MSE
from npann.Optimizers.RMSProp import RMSProp
import cv2
import time


#X, y = loadMNIST()
X = loadFaces()
# X = loadWatches()
# X = loadDogs()

number_epochs = 1000000
accuracy_report_freq = 5
minibatch_size = 64 #len(X)
dataset_size = len(X)
vector_size = len(X[0])
magnification = 10
image_size = int(math.sqrt(vector_size))  # ONLY SQUARE IMAGES!!!!


lr = 0.001
number_hidden = 35
discriminator = Sequential()
discriminator.addLayer(Dense(vector_size, number_hidden))
discriminator.addLayer(Sigmoid())
discriminator.addLayer(Dense(number_hidden, 1))
discriminator.addLayer(Sigmoid())
discriminator.addLoss(MSE())
discriminator.addOptimizer(RMSProp(learning_rate = lr))

lr = 0.001
number_hidden = 35
generator = Sequential()
generator.addLayer(Dense(vector_size, number_hidden))
generator.addLayer(Sigmoid())
generator.addLayer(Dense(number_hidden, vector_size))
generator.addLayer(Sigmoid())
generator.addLoss(MSE())
generator.addOptimizer(RMSProp(learning_rate = lr))



seeds = np.random.rand(25, vector_size)

def generateImage(arr):
    img = np.reshape(arr, (image_size, image_size))
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def visualizeOne():
    arr = seeds[0]
    arr = generator.forward(arr)
    res = generateImage(arr)
    cv2.imshow('Generated Image', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

    # time.sleep(.1)

mse = MSE()

for epoch in range(number_epochs):
    all_minibatch_indexes = np.random.permutation(dataset_size)
    for j in range(dataset_size / minibatch_size):
        minibatch_start = j * minibatch_size
        minibatch_end = (j + 1) * minibatch_size
        minibatch_indexes = all_minibatch_indexes[minibatch_start:minibatch_end]
        minibatch_X = X[minibatch_indexes]
        # minibatch_y = y[minibatch_indexes]

        # Iterate using real data (targets for discriminator are 1's)
        # and using fake, generated data (targets for discriminator are 0's)
        minibatch_y_ones = np.ones((minibatch_X.shape[0], 1))
        random_minibatch = np.random.rand(minibatch_size, vector_size)
        generator_output = generator.forward(random_minibatch)
        minibatch_y_zeros = np.zeros((minibatch_X.shape[0], 1))

        train_X = np.vstack([minibatch_X, generator_output])
        train_y = np.vstack([minibatch_y_ones, minibatch_y_zeros])
        minibatch_err = discriminator.iterate(train_X, train_y)

        # Iterate for generator using discriminator's outputs
        random_minibatch = np.random.rand(minibatch_size, vector_size)
        generator_output = generator.forward(random_minibatch)
        discriminator_output = discriminator.forward(generator_output)

        minibatch_y_ones = np.ones((minibatch_X.shape[0], 1))
        grad = mse.calculateGrad(discriminator_output, minibatch_y_ones)
        discriminator_grad = discriminator.backward(grad)

        generator.backward(discriminator_grad)
        generator.update()
        loss = mse.calculateLoss(discriminator_output, minibatch_y_ones)
        minibatch_err3 = np.linalg.norm(loss)

        if epoch % 10 == 0:
            print "Epoch %6d: %.8f %.8f" % (epoch, minibatch_err, minibatch_err3)

        # Visualize the generated image for seed value
        visualizeOne()
