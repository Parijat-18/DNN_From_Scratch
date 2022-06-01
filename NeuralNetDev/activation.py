import numpy as np

class Activation:
    def sigmoid(self , Z):
       
        s = 1 / (1 + np.exp(-1 * Z))
        return s

    def relu(self , Z):
        s = np.maximum(0 , Z)
        assert(s.shape == Z.shape)
        return s
    
    def tanh(self , Z):
        s = np.tanh(Z)
        assert(s.shape == Z.shape)
        return s
    
    def softmax(self , Z):
        t = np.exp(Z)
        t_sum = np.sum(t)
        s = t / t_sum
        assert(s.shape == Z.shape)
        return s
    
    def hardmax(self , Z):
        t = np.exp(Z)
        t_sum = np.sum(t)
        s = t / t_sum
        s = (np.amax(s) == s).astype('int')
        return s

    def sigmoid_backward(self , dA , cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        assert(dZ.shape == Z.shape)
        return dZ
    
    def relu_backward(self , dA , cache):
        Z = cache
        dZ = np.array(dA , copy=True)
        dZ[Z <= 0] = 0
        assert(dZ.shape == Z.shape)
        return dZ
    
    def tanh_backward(self , dA , cache):
        Z = cache
        s = np.tanh(Z)
        dZ = dA * (1 - (s * s))
        assert(dZ.shape == Z.shape)
        return dZ
    
    def softmax_backward(self , A , Y):
        assert(A.shape == Y.shape)
        dZ = A - Y
        return dZ