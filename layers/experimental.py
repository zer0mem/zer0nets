import numpy as np
from layers.feat_space import *

class AntiSaturationSig(FeatureSpace):
    def __init__(self, degree, index):
        super(AntiSaturationSig, self).__init__(degree, degree, index)
        self.w = np.identity(self.w.shape[0])

    def name(self):
        return "ass ~ sigmoid anti saturating layer"

    def signal(self, x):
        saturated_part = x > .95
        saturated_part += x < .05
        return (saturated_part == False) * x
        ##  saturated_part = saturated_part == False
        #return 1 * np.cos(saturated_part * x * 1000) + (saturated_part == False) * x * 100

    def prime(self, x):
        saturated_part = x > .95
        saturated_part += x < .05
        return (saturated_part == False)
        ##  saturated_part = saturated_part == False
        #return 1 * np.sin(saturated_part * x * 1000) + (saturated_part == False) * 100

    def __calc_err__(self, y_predicted, y):
        assert False, "not intended to be last layer!"
    def loss(self, y_predicted, y):#None
        assert False, "cost function not implemented"
