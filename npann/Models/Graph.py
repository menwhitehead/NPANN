import numpy as np
from npann.Utilities.misc_functions import accuracy, accuracyBinary

class Graph:

    def __init__(self):
        self.layers = {}  # each layer has a name
        self.backward_connections = {} # links from key back to value (both layer names)
        self.forward_connections = {} # links from key forward to value (both layer names)
        self.output_layers = [] # names of layers that output from network
        self.input_layers = {} # names of inputs

    def addLayer(self, layer_name, layer, inputs, is_output=False):
        self.layers[layer_name] = layer
        self.backward_connections[layer_name] = []
        if is_output:
            self.output_layers.append(layer_name)

        for incoming_layer in inputs:
            if incoming_layer not in self.forward_connections:
                self.forward_connections[incoming_layer] = []
            self.forward_connections[incoming_layer].append(layer_name)
            self.backward_connections[layer_name].append(incoming_layer)


    def addLoss(self, loss_layer):
        self.loss_layer = loss_layer

    def addOptimizer(self, opt):
        self.optimizer = opt

    def __getitem__(self, layer_name):
        return self.layers[layer_name]

    def setInput(self, input_name, X):
        self.input_layers[input_name] = X

    def computationComplete(self, layers_computed):
        for layer_name in layers_computed:
            if layers_computed[layer_name] == False:
                return False
        return True

    def forwardLayerInputsReady(self, layer_name, layers_computed):
        for incoming_layer in self.backward_connections[layer_name]:
            if not layers_computed[incoming_layer]:
                return False
        return True

    def backwardLayerInputsReady(self, layer_name, layers_computed):
        if layer_name not in self.forward_connections:
            return True
        for incoming_layer in self.forward_connections[layer_name]:
            if not layers_computed[incoming_layer]:
                return False
        return True


    def forward(self, named_input_pairs, train=True):  # pairs of (name, X) for inputs
        # Set up the inputs to the entire Graph
        for input_name in named_input_pairs:
            self.setInput(input_name, named_input_pairs[input_name])

        layer_outputs = {}

        # Init all layers to need to be computed
        layers_computed = {}
        for layer_name in self.layers:
            layers_computed[layer_name] = False
        for layer_name in self.input_layers:
            layers_computed[layer_name] = True
            layer_outputs[layer_name] = self.input_layers[layer_name]

        while not self.computationComplete(layers_computed):
            for layer_name in self.layers:
                if not layers_computed[layer_name]:
                    if self.forwardLayerInputsReady(layer_name, layers_computed):
                        if self.layers[layer_name].__class__.__name__ != "Merge":
                            incoming_layer = self.backward_connections[layer_name][0]
                            layer_outputs[layer_name] = self.layers[layer_name].forward(layer_outputs[incoming_layer])
                            layers_computed[layer_name] = True
                        else:
                            # Treat a merging layer differently
                            merge_inputs = []
                            for incoming_layer in self.backward_connections[layer_name]:
                                merge_inputs.append(layer_outputs[incoming_layer])
                            layer_outputs[layer_name] = self.layers[layer_name].forward(merge_inputs)
                            layers_computed[layer_name] = True

        # Should output multiple values in the future?
        # for now only output a single value...
        for layer_name in self.output_layers:
            return layer_outputs[layer_name]

        return "ERROR!!!"



    def backward(self, loss_grad):

        minibatch_size = len(loss_grad)

        # Again, assume a single output layer for now...
        curr_layer = self.output_layers[0]

        layer_outputs = {}

        # Init all layers to need to be computed
        layers_computed = {}
        for layer_name in self.layers:
            layers_computed[layer_name] = False

        while not self.computationComplete(layers_computed):
            for layer_name in self.layers:
                ly = self.layers[layer_name]
                if not layers_computed[layer_name]:
                    if self.backwardLayerInputsReady(layer_name, layers_computed):

                        if layer_name in self.output_layers:
                            inc_grad = loss_grad
                        else:
                            if ly.__class__.__name__ == "Merge":
                                layer_size = ly.number_connections

                            elif hasattr(ly, 'weights'):
                                layer_size = ly.weights.shape[1]
                            else:
                                layer_size = ly.incoming_acts.shape[1]

                            inc_grad = np.zeros((minibatch_size, layer_size))

                            for other_layer in self.forward_connections[layer_name]:
                                if self.layers[other_layer].__class__.__name__ != "Merge":
                                    inc_grad += layer_outputs[other_layer]
                                else:
                                    # need to extract out this layer's part of the merge's gradient
                                    merge_grad = layer_outputs[other_layer]
                                    this_layer_index = self.backward_connections[other_layer].index(layer_name)
                                    start, end = this_layer_index * layer_size, (this_layer_index + 1) * layer_size
                                    inc_grad += merge_grad[:,start:end]

                        layer_outputs[layer_name] = self.layers[layer_name].backward(inc_grad)
                        layers_computed[layer_name] = True

        return "BACKPROP DONE!!!"


    def update(self):
        for layer_name in self.layers:
            self.layers[layer_name].update(self.optimizer)

    def iterate(self, X, y):
        output = self.forward(X)
        self.loss = self.loss_layer.calculateLoss(output, y)
        curr_grad = self.loss_layer.calculateGrad(output, y)
        final_grad = self.backward(curr_grad)
        self.update()
        return np.linalg.norm(self.loss)

    def resetLayers(self):
        for layer in self.layers:
            layer.reset()

    def train(self, X, y, minibatch_size, number_epochs, verbose=0):
        #self.resetLayers()  # clear out any old tables/state
        dataset_size = len(X)
        for i in range(number_epochs):
            all_minibatch_indexes = np.random.permutation(dataset_size)
            epoch_err = 0
            for j in range(dataset_size / minibatch_size):
                minibatch_indexes = all_minibatch_indexes[j * minibatch_size:j * minibatch_size + minibatch_size]
                minibatch_X = X[minibatch_indexes]
                minibatch_y = y[minibatch_indexes]
                inputs = {"input1": minibatch_X}
                minibatch_err = self.iterate(inputs, minibatch_y)
                epoch_err += minibatch_err

            if verbose==1:
                print "Epoch #%d, Error: %.8f" % (i, epoch_err)
            elif verbose==2:
                print "Epoch #%d, Error: %.8f, Accuracy: %.4f" % (i, epoch_err, accuracyBinary(self, X, y))



    def __str__(self):
        result = ''
        for layer in self.layers:
            result += str(layer.weights) + "\n"

        return result
