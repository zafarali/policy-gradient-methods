import numpy as np
import torch
from torch.autograd import Variable

# used to interface between pytorch and gym/other environments
class Interface(object):
    pass


def list2pytorch(tuple_):
    """
    Converts the gym tuple into a torch variable of size (1, -1)
    """
    return Variable(torch.from_numpy(np.array(tuple_).reshape(1, -1)))

def pytorch2array(tensor):
    """
    Converts a pytorch Tensor or Variable of size (1, -1)
    into a flat list 
    """
    if type(tensor) is Variable:
        return tensor.data.numpy().reshape(-1)
    elif type(tensor) is torch.Tensor:
        return tensor.numpy().reshape(-1)
    else:
        return tensor

def pytorch2list(tensor):
    return pytorch2array(tensor).tolist()

gym2pytorch = list2pytorch