import numpy as np
from misc_functions import accuracy, accuracyBinary
from Sequential import Sequential

class SequentialReinforcement(Sequential):

    def iterate(self, X, y):
        output = self.forward(X)

        reward = self.loss_layer.calculateLoss(output, y)
        for layer in self.layers:
            if layer.__class__.__name__ == "Reinforce":
                # print "MATCH", layer

                layer.reward = 1 - (np.abs(reward)/np.max(reward))

        final_grad = self.backward(output, y)
        self.update()
        return np.linalg.norm(self.loss)
