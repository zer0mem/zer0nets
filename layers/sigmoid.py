import numpy as np
from layers.feat_space import *

class Sigmoid(FeatureSpace):
    def name(self):
        return "sigmoid"
    def signal(self, x):
        return 1 / (1 + np.exp(-x))
    def prime(self, x):
        return x * (1. - x)
    def __calc_err__(self, y_predicted, y):
        return -1. * (np.divide(y, y_predicted) - np.divide(1 - y, 1 - y_predicted))
    def loss(self, y_predicted, y):#cross-entropy
        return np.mean(np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted)))
