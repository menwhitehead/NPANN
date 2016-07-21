  
    
class CategoricalCrossEntropy:
    
    def calculateLoss(self, output, target):
        #return (target * np.log(output) + (1 - target) * np.log(1 - output))
        return target - output
    
    def calculateGrad(self, output, target):
        return target - output
