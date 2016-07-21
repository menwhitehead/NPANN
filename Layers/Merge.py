
class Merge(Layer):
    
    def forward(self, list_inputs):
        return np.stack(list_inputs)
    
    def backward(self, grad):
        return grad
