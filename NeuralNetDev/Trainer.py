import copy
import math
import numpy as np
from IPython.display import clear_output 
from NeuralNetDev.prop_algo import forward_prop , backward_prop

class l_layer_dnn_model:
    def __init__(self , X , Y , layer_dims, learning_rate, iterations, mini_batch_size=None, optimizer=None , lambd= 0, dropt= None, validation_data=None):
        self.iterations = iterations
        self.validation_data = validation_data
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.dropt = dropt
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)
        self.layer_dims = layer_dims
        self.layer_dims = [X.shape[0]] + self.layer_dims
        print(self.layer_dims)
        self.m = X.shape[1]
        self.parameters = self.initialize_params()

    def initialize_params(self):
        parameters = {}
        l = len(self.layer_dims)
        for i in range(1 , l):
            parameters["W" + str(i)] = np.random.randn(self.layer_dims[i] , self.layer_dims[i - 1]) * np.sqrt(2/self.layer_dims[i - 1])
            parameters["b" + str(i)] = np.zeros((self.layer_dims[i] , 1))
        return parameters

    
    def min_batch_selection(self , mini_batch_size):

        perm = list(np.random.permutation(self.m))
        X_shuffled = self.X[: , perm]
        Y_shuffled = self.Y[: , perm].reshape(self.Y.shape[0] , self.m)
        mini_batch = []

        num_mini_batches = math.floor(self.m / mini_batch_size)
        for i in range(0 , num_mini_batches):
            X_batch = X_shuffled[: , i*mini_batch_size:(i+1)*mini_batch_size]
            Y_batch = Y_shuffled[: , i*mini_batch_size:(i+1)*mini_batch_size]
            mini_batch.append((X_batch , Y_batch))

        if self.m % mini_batch_size != 0:
            X_batch = X_shuffled[: , (num_mini_batches*mini_batch_size):]
            Y_batch = Y_shuffled[: , (num_mini_batches*mini_batch_size):]
        
        mini_batch.append((X_batch , Y_batch))

        return mini_batch

    def compute_cost(self , AL , Y):
        L = len(self.parameters) // 2
        L2_reg_cost = 0
        cost = (-1 / self.m) * np.sum(np.multiply(Y , np.log(AL)) + np.multiply((1 - Y) , np.log(1 - AL)))
        for l in range(L):
            L2_reg_cost += np.sum(np.square(self.parameters['W' + str(l+1)]))
        L2_reg_cost = (1/2) * (self.lambd/self.m) * L2_reg_cost

        cost += L2_reg_cost
        cost = np.squeeze(cost)
        return cost

    def update_parameters(self , grads , t , model=None):
        if model == None:
            L = len(self.parameters) // 2
            for l in range(1 , L+1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate*grads["dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate*grads["db" + str(l)]
        else:
            self.parameters = model.optimize(grads , self.parameters , t)

    def predict(self , X , to_numerical=False):
        dropt = [1 for _ in range(len(self.parameters) // 2)]
        forward = forward_prop()
        AL , _ , _ = forward.L_model_forward(X , self.parameters , dropt)
        if self.Y.shape[0] > 1:
            if to_numerical: predictions = np.argmax(AL , axis=0)
            else: predictions = (AL == np.amax(AL , axis=0)).astype('int')
        else:
            predictions = np.rint(AL)
        
        return predictions
    
    def fit(self , X , Y , output , batch):
        forward = forward_prop()
        backward = backward_prop()
        for i in range(0 , self.iterations):
            AL , caches , D = forward.L_model_forward(X , self.parameters , self.dropt)
            cost = self.compute_cost(AL , Y)
            grads = backward.L_model_backward(AL , Y , caches , self.dropt , D , self.lambd)
            self.update_parameters(grads , i+1 , self.optimizer)
            train_accuracy = 100 - np.mean(np.abs(self.predict(X) - Y))*100
            if self.validation_data != None:
                predict_test_y = self.predict(self.validation_data[0])
                test_accuracy = 100 - np.mean(np.abs(predict_test_y - self.validation_data[1]))*100
                self.test_cost.append(100.0 - test_accuracy)
            if self.validation_data != None:
                print(f"{i+1}\{self.iterations}, batch= {batch}, train-accuracy: {100 - (cost*100)} , test-accuracy: {test_accuracy}\n")
            else:
                print(f"{i+1}\{self.iterations}, batch= {batch}, train-accuracy: {100 - (cost*100)}\n")
            self.train_cost.append(cost*100)
        if self.validation_data != None:
            output.append(f"{i+1}\{self.iterations}, batch= {batch}, train-accuracy: {100 - (cost*100)} , test-accuracy: {test_accuracy}\n")
        else:
            output.append(f"{i+1}\{self.iterations}, batch= {batch}, train-accuracy: {100 - (cost*100)}\n")
        return output
    
    def train(self):
        self.train_cost = []
        self.test_cost = []
        output = []
        if self.mini_batch_size == None:
            output = self.fit(self.X , self.Y , output , "...")
            print(''.join(map(str , output)))
        else:
            mini_batches = self.min_batch_selection(self.mini_batch_size)
            batch = 1
            for X , Y , in mini_batches:
                output = self.fit(X , Y , output , batch)
                clear_output(wait=True)
                batch += 1