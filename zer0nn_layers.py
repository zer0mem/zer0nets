from layers.input import *
from layers.linear import Linear
from layers.sigmoid import Sigmoid
from layers.relu import ReLU
from layers.experimental import AntiSaturationSig

class SpaceLayerFactory:
    def layer(name, dim1, dim2, ind):
        if "input" == name:
            assert False, "input layer must be add-hot pluged to neural net at forward pass per new input!"
        if "lin" == name:
            return Linear(dim1, dim2, ind)
        if "sig" == name:
            return Sigmoid(dim1, dim2, ind)
        if "relu" == name:
            return ReLU(dim1, dim2, ind)
        if "ass" == name:
            return AntiSaturationSig(dim1, ind)
        assert False, "space-layer : <%s> not implemented!"%name
        return None
