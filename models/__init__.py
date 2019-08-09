from models.onestream import OneStream
from models.twostream import TwoStream
from models.physicsmodel import rgb_nir_dcp
from models.resphysics import ResidualPhysics

def get_model(model):
    if model == 'residual_physics':
        net = ResidualPhysics('resnet')
    elif model == 'one_stream':
        net = OneStream()
    elif model == 'two_stream':
        net = TwoStream()
    elif model == 'dehazenet':
        pass
    elif model == 'our_model':
        pass
    return net
