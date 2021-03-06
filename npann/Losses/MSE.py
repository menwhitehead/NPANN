import numpy as np

class MSE:

    def calculateLoss(self, output, target):
        return np.power(target - output, 2) #/ len(output)

    def calculateGrad(self, output, target):
        return 2 * (target - output)
