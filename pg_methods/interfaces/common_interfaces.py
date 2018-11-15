import numpy as np
import torch

# used to interface between pytorch and gym/other environments
class Interface(object):
    def __init__(self):
        pass

    def gym2pytorch(self, gym_representation):
        """
        Converts the gym representation to pytorch
        :param gym_representation:
        :return:
        """
        raise NotImplementedError

    def pytorch2gym(self, pytorch_representation):
        """
        Converts pytorch representation to gym.
        :param pytorch_representation:
        :return:
        """
        raise NotImplementedError


def number_convert(number):
    """
    Converts a np.dtype into corresponding python type
    :param number:
    :return:
    """
    if type(number) in [np.int64, np.int32]:
        return int(number)
    if type(number) in [np.float32, np.float64]:
        return float(number)
    else:
        return number

NUMBERS = [np.int64, np.int32, np.float32, np.float64, int, float]

def list2pytorch(tuple_):
    """
    Converts the gym tuple into a torch variable of size (1, -1)
    """
    return torch.from_numpy(np.array(tuple_).reshape(1, -1))

def pytorch2array(tensor):
    """
    Converts a pytorch Tensor or Variable of size (1, -1)
    into a flat list 
    """
    if type(tensor) is torch.Tensor:
        return tensor.numpy().reshape(-1)
    else:
        return tensor

def pytorch2list(tensor):
    return pytorch2array(tensor).tolist()

gym2pytorch = list2pytorch