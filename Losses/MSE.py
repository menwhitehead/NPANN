import gnumpy as np
import numpy as np2

class MSE:
    
    def calculateLoss(self, output, target):
        return np2.power(target - output, 2) #/ len(output)
    
    def calculateGrad(self, output, target):
        return target - output