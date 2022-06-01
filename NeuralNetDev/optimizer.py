import numpy as np

class Optimizer:
    class RMSProp:

        def __init__(self , learning_rate , beta=0.999 , epsilon= 1e-8):
            self.beta = beta
            self.epsilon = epsilon
            self.learning_rate = learning_rate
        
        def initializer(self , parameters):
            L = len(parameters) // 2
            s = {}
            for l in range(1 , L+1):
                s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
            return s

        def optimize(self , grads , parameters , t):
            L = len(parameters) // 2
            s = self.initializer(parameters)
            s_corr = {}
            
            for l in range(1 , L+1):
                s["dW" + str(l)] = (self.beta * s["dW" + str(l)]) + ((1 - self.beta)*np.multiply(grads["dW" + str(l)] , grads["dW" + str(l)]))
                s["db" + str(l)] = (self.beta * s["db" + str(l)]) + ((1 - self.beta)*np.multiply(grads["db" + str(l)] , grads["db" + str(l)]))
                
                s_corr["dW" + str(l)] = s["dW" + str(l)] / (1 - pow(self.beta , t)) 
                s_corr["db" + str(l)] = s["db" + str(l)] / (1 - pow(self.beta , t)) 

                parameters["W" + str(l)] = parameters["W" + str(l)] - (self.learning_rate*np.divide(grads["dW" + str(l)] , (np.sqrt(s_corr["dW" + str(l)]) + self.epsilon)))
                parameters["b" + str(l)] = parameters["b" + str(l)] - (self.learning_rate*np.divide(grads["db" + str(l)] , (np.sqrt(s_corr["db" + str(l)]) + self.epsilon)))

            return parameters

    class momentum:
        def __init__(self , learning_rate , beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta

        def initializer(self , parameters):
            L = len(parameters) // 2
            s = {}
            for l in range(1 , L+1):
                s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
            return s

        def optimize(self , grads , parameters , t):
            L = len(parameters) // 2
            v = self.initializer(parameters)
            v_corr = {}
            
            for l in range(1 , L+1):
                v["dW" + str(l)] = (self.beta * v["dW" + str(l)]) + ((1 - self.beta)*grads["dW" + str(l)])
                v["db" + str(l)] = (self.beta * v["db" + str(l)]) + ((1 - self.beta)*grads["db" + str(l)])
                
                v_corr["dW" + str(l)] = v["dW" + str(l)] / (1 - pow(self.beta , t)) 
                v_corr["db" + str(l)] = v["db" + str(l)] / (1 - pow(self.beta , t)) 

                parameters["W" + str(l)] = parameters["W" + str(l)] - (self.learning_rate*v["dW" + str(l)])
                parameters["b" + str(l)] = parameters["b" + str(l)] - (self.learning_rate*v["db" + str(l)])

            return parameters
        



    class Adam:
        def __init__(self , learning_rate , beta1 = 0.9 , beta2 = 0.999 , epsilon= 1e-8):
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.learning_rate = learning_rate
        
        def initializer(self , parameters):
            L = len(parameters) // 2
            v = {}
            s = {}
            for l in range(1 , L+1):
                v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
                s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
                s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
            return v , s

        def optimize(self , grads , parameters , t):
            L = len(parameters) // 2
            v , s = self.initializer(parameters)
            v_corr = {}
            s_corr = {}
            for l in range(1 , L+1):
                v["dW" + str(l)] = (self.beta1 * v["dW" + str(l)]) + ((1 - self.beta1)*grads["dW" + str(l)])
                v["db" + str(l)] = (self.beta1 * v["db" + str(l)]) + ((1 - self.beta1)*grads["db" + str(l)])
                s["dW" + str(l)] = (self.beta2 * s["dW" + str(l)]) + ((1 - self.beta2)*np.multiply(grads["dW" + str(l)] , grads["dW" + str(l)]))
                s["db" + str(l)] = (self.beta2 * s["db" + str(l)]) + ((1 - self.beta2)*np.multiply(grads["db" + str(l)] , grads["db" + str(l)]))
                
                v_corr["dW" + str(l)] = v["dW" + str(l)] / (1 - pow(self.beta1 , t)) 
                v_corr["db" + str(l)] = v["db" + str(l)] / (1 - pow(self.beta1 , t)) 
                s_corr["dW" + str(l)] = s["dW" + str(l)] / (1 - pow(self.beta2 , t)) 
                s_corr["db" + str(l)] = s["db" + str(l)] / (1 - pow(self.beta2 , t)) 

                parameters["W" + str(l)] = parameters["W" + str(l)] - (self.learning_rate*np.divide(v_corr["dW" + str(l)] , (np.sqrt(s_corr["dW" + str(l)]) + self.epsilon)))
                parameters["b" + str(l)] = parameters["b" + str(l)] - (self.learning_rate*np.divide(v_corr["db" + str(l)] , (np.sqrt(s_corr["db" + str(l)]) + self.epsilon)))

            return parameters