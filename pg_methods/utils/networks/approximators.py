import torch
import torch.nn as nn
from collections import OrderedDict

def MLP_factory(input_size,
                hidden_sizes=[],
                output_size=1,
                hidden_non_linearity=None,
                out_non_linearity=None):
    """
    Creates simple MLPs
    :param input_size: an int for the size of the input
    :param hidden_sizes: list of hidden sizes for each layer
    :param output_size: an int for the size of the output
    :param non_linearity: the non-linearity to use (must be a nn.Module)
    """
    # if type(hidden_sizes) is int:
    #     hidden_sizes = [ hidden_sizes for _ in range(n_layers) ]    
    
    assert type(hidden_sizes) is list
    layers = []

    if len(hidden_sizes) == 0:
        # no hidden layers
        layers.append(('MLP', nn.Linear(input_size, output_size)))
    else:
        layers.append(('Input', nn.Linear(input_size, hidden_sizes[0])))
        
        for i in range(len(hidden_sizes)-1):
            layers.append(
                ('linear_{}'.format(i),
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            )
            if hidden_non_linearity is not None:
                layers.append(
                    ('h_act_{}_{}'.format(i, hidden_non_linearity.__name__),
                    hidden_non_linearity())
                )

        layers.append(('Out', nn.Linear(hidden_sizes[-1], output_size)))
    
    if out_non_linearity is not None:
        layers.append(
            ('out_act_{}'.format(out_non_linearity.__name__),
            out_non_linearity())
        )
    return nn.Sequential(OrderedDict(layers))

