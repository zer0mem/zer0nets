from layers.feat_space import *

class Input(FeatureSpace):
    def signal(self, _):
        assert False, "input layer do not emit any signal!"
    def prime(self, _):
        assert False, "input layer do not backward any error!"
    def __calc_err__(self, y_predicted, y):
        assert False, "input layer has no error function!"
    def loss(self, y_predicted, y):#mse
        return np.mean(self.__calc_err__(y_predicted, y) ** 2)

    def __init__(self, name, X):
        self.w = X
        self.name = name
    def signal_out(self):
        return self.w
    def name(self):
        return self.name

