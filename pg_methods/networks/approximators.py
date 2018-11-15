from collections import OrderedDict

import torch.nn as nn

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
    
    # assert type(hidden_sizes) is list
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

class TwoHeadNetwork(nn.Module):
    def __init__(self, shared_body, head_in, head_out):
        super().__init__()
        self.shared_body = shared_body
        self.first_head = nn.Linear(head_in, head_out)
        self.second_head = nn.Linear(head_in, head_out)

    def forward(self, x):
        intermediate_representation = self.shared_body(x)
        first_out = self.first_head(intermediate_representation)
        second_out = self.second_head(intermediate_representation)

        return first_out, second_out

def MLP_factory_two_heads(input_size,
                          hidden_sizes=[],
                          output_size=1,
                          hidden_non_linearity=None,
                          out_non_linearity=None):
    """
    Creates a neural network with two heads and a shared body
    :param input_size:
    :param hidden_sizes:
    :param output_size:
    :param hidden_non_linearity:
    :param out_non_linearity:
    :return:
    """
    shared_body = MLP_factory(input_size,
                              hidden_sizes[:-1],
                              output_size=hidden_sizes[-1],
                              hidden_non_linearity=hidden_non_linearity,
                              out_non_linearity=out_non_linearity)

    return TwoHeadNetwork(shared_body, hidden_sizes[-1], output_size)

