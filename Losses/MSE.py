import numpy as np  
    
class MSE:
    
    def calculateLoss(self, output, target):
        return np.power(target - output, 2)
    
    def calculateGrad(self, output, target):
        return target - output