import numpy as np
from layers.feat_space import *

class Linear(FeatureSpace):
    def name(self):
        return "linear"
    def signal(self, x):
        return x
    def prime(self, _):
        return 1.
    def __calc_err__(self, y_predicted, y):
        return y - y_predicted
    def loss(self, y_predicted, y):#mse
        return np.mean(self.__calc_err__(y_predicted, y) ** 2)

