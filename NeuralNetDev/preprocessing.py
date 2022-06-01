from cv2 import sqrt
import numpy as np

class Normalizer:
    def __init__(self , X , epsilon=1e-8) -> None:
        assert(X.ndim == 2)
        self.m = X.shape[1]
        self.epsilon = epsilon
        self.mean = (1/self.m) * np.sum(X)
        self.variance = (1 / self.m) * np.sum(np.square(X - self.mean))
    
    def normalize(self , X):
        X_norm = (X - self.mean)
        print(self.variance)
        X_norm = (1 / (np.sqrt(self.variance) + self.epsilon)) * X_norm
        return X_norm