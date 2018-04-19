from layers.feat_space import *

class ReLU(FeatureSpace):
    def name(self):
        return "ReLU"

    def signal(self, x):
        saturated_part = x > .0
        return saturated_part * x

    def prime(self, x):
        saturated_part = x > .0
        return saturated_part

    def __calc_err__(self, y_predicted, y):
        return y - y_predicted

    def loss(self, y_predicted, y):#mse
        return np.mean(self.__calc_err__(y_predicted, y) ** 2)
