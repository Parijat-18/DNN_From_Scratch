import numpy as np
from NeuralNetDev.activation import Activation

class DropOut:
    def Dropout(self , A , keep_probs):
        D = np.random.rand(A.shape[0] , A.shape[1])
        D = (D < keep_probs).astype('int')
        A = np.multiply(A , D)
        A = A / keep_probs
        return A , D

class forward_prop:
    
    def linear_forward(self , A , W , b):
        Z = np.dot(W , A) + b
        cache = ((A , W , b) , Z)
        return Z , cache

    def linear_activation_forward(self , A_prev , W , b, activation):
        actv_fnc = Activation()
        Z , cache = self.linear_forward(A_prev , W , b)
        if activation == "sigmoid":
            A = actv_fnc.sigmoid(Z)
        elif activation == 'tanh':
            A = actv_fnc.tanh(Z)
        elif activation == "relu":
            A = actv_fnc.relu(Z)
        elif activation == "softmax":
            A = actv_fnc.softmax(Z)
        return A , cache
    
    def L_model_forward(self , X , parameters, dropt):
        caches = []
        D = []
        A = X
        if dropt != None: dropout_fnc = DropOut()
        L = len(parameters) // 2
        for l in range(1 , L):
            A_prev = A
            A , cache = self.linear_activation_forward(A_prev , parameters["W" + str(l)] , parameters["b" + str(l)] , "tanh")
            if dropt != None:
                A , d = dropout_fnc.Dropout(A , dropt[l - 1])
                D.append(d)
            caches.append(cache)
        AL , cache = self.linear_activation_forward(A , parameters["W" + str(L)] , parameters["b" + str(L)] , "sigmoid")
        caches.append(cache)
        if dropt != None:
            return AL , caches , D
        return AL , caches , None

class backward_prop:
    def linear_backward(self , dZ , m , lambd ,  cache):
        A_prev , W , b = cache
        dW = (1 / m) * np.dot(dZ , A_prev.T) + ((lambd / m) * W)
        db = (1 / m) * np.sum(dZ , axis=1 , keepdims=True)
        dA_prev = np.dot(W.T , dZ)
        return dA_prev , dW , db
    
    def linear_activation_backward(self , dA , AL , Y , caches , m , lambd , activation):
        linear_cache , activation_cache = caches
        actv_fnc = Activation()

        if activation == "relu":
            dZ = actv_fnc.relu_backward(dA , activation_cache)
        elif activation == "tanh":
            dZ = actv_fnc.tanh_backward(dA , activation_cache)
        elif activation == "sigmoid":
            dZ = actv_fnc.sigmoid_backward(dA , activation_cache)
        elif activation == "softmax":
            dZ = actv_fnc.softmax_backward(AL , Y)
        
        dA_prev , dW , db = self.linear_backward(dZ , m  , lambd , linear_cache)
        return dA_prev , dW , db

    def L_model_backward(self , AL , Y , caches , keep_prob , D= None , lambd= 0):
        grads = {}
        L = len(caches)
        current_cache = caches[L - 1]
        Y = Y.reshape(AL.shape)
        m = Y.shape[1]
        dAL = -1 * (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev_tmp , dW , db = self.linear_activation_backward(dAL , AL , Y , current_cache , m , lambd , "sigmoid")
        grads["dA" + str(L - 1)] = dA_prev_tmp
        grads["dW" + str(L)] = dW
        grads["db" + str(L)] = db
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            if D != None:
                grads["dA" + str(l + 1)] = np.multiply(grads["dA" + str(l + 1)] , D[l])
                grads["dA" + str(l + 1)] = grads["dA" + str(l + 1)] / keep_prob[l]
            dA_prev_tmp , dW , db = self.linear_activation_backward(grads["dA" + str(l + 1)] , AL , Y , current_cache , m , lambd , "tanh")
            grads["dA" + str(l)] = dA_prev_tmp
            grads["dW" + str(l + 1)] = dW
            grads["db" + str(l + 1)] = db
        return grads
    