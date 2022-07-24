import torch
import torch.nn as nn
from .NARXNET import NARXNET

class NARX(NARXNET):
    """Implementation of NN-NARX
    """    
    def __init__(self, output_lags: int, input_lags: list[int], n_hidden = 3, layersize = 10, afunc = torch.relu, bias = True):
        """

        Args:
            output_lags (int): number of output lags
            input_lags (list[int]): list of numbers of input lags
            n_hidden (int, optional): number of hidden layers of each subnet. Defaults to 2.
            layersize (int, optional): size of hidden layers of each subnet. Defaults to 10.
            afunc (_type_, optional): activation function used in subnets. Defaults to torch.relu.
            bias (bool, optional): whether Linear layers have a bias or not. Defaults to True.
        """        
        assert n_hidden > 1
        super(NARX, self).__init__(output_lags, input_lags)
        self.afunc = afunc
        self.bias = bias
        self.n_netinputs = output_lags + sum(input_lags)
        self.linear_layers = [nn.Linear(layersize, layersize, bias = self.bias) for _ in range(n_hidden)]
        self.linear_layers[0] = nn.Linear(self.n_netinputs, layersize, bias = self.bias)
        self.linear_layers[-1] = nn.Linear(layersize, 1, bias = self.bias)
        self.linear_layers = nn.ModuleList(self.linear_layers)

    
    def forward(self, output_lagged: torch.Tensor, inputs_lagged: list[torch.Tensor]):
        """ Compute prediction of NARX-Model
        This works with single step and with batch data
        Args:
            output_lagged (torch.Tensor): Lagged Outputs; shape should be [output_lags] 
            inputs_lagged (list[torch.Tensor]): Lagged Inputs; each shape should be [input_lags[i]]

        Returns:
            torch.Tensor: Output. Shape should be [1]
        """
        netinputlist = inputs_lagged
        netinputlist.append(output_lagged)
        x = torch.cat(netinputlist, dim = 1)
        for layer in self.linear_layers[:-1]:
            x = self.afunc(layer(x))
        return self.linear_layers[-1](x)