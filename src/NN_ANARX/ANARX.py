import torch
import torch.nn as nn
from .NARXNET import NARXNET


class ANARX(NARXNET):
    def __init__(self, output_lags: int, input_lags: list[int], n_hidden = 2, layersize = 10, afunc = torch.relu, bias = True, SANARX = False):
        """ 
        Args:
            output_lags (int): number of output lags
            input_lags (list[int]): list of numbers of input lags
            n_hidden (int, optional): number of hidden layers of each subnet. Defaults to 2.
            layersize (int, optional): size of hidden layers of each subnet. Defaults to 10.
            afunc (_type_, optional): activation function used in subnets. Defaults to torch.relu.
            bias (bool, optional): whether Linear layers have a bias or not. Defaults to True.
            SANARX (bool, optional): wheter or not the Model is in SANARX-Form. SANARX-Means the first subnet consists of only a Linear layer. Defaults to False.
        """        
        assert n_hidden > 1
        super(ANARX, self).__init__(output_lags, input_lags)
        self.n_subnets = max(max(input_lags), output_lags)
        
        # Construct subnets according to the chosen form 
        self.lag_map = {}
        netinputs = input_lags
        netinputs.append(output_lags)
        n_netinputs = len(netinputs)
        for i in range(self.n_subnets):
            mask = [0]*n_netinputs
            for j in range(n_netinputs):
                if netinputs[j] > i:
                    mask[j] = 1
            self.lag_map[i] = mask
        self.subnets = [LAGNET(sum(self.lag_map[i]), n_hidden, layersize, afunc, bias) for i in range(self.n_subnets)]
        if SANARX:
            id = self.identity
            self.subnets[0] = LAGNET(sum(self.lag_map[0]), 1, sum(self.lag_map[0]), id, bias)
        self.subnets = nn.ModuleList(self.subnets)
    
    def forward(self, output_lagged: torch.Tensor, inputs_lagged: list[torch.Tensor]):
        """ Compute prediction of ANARX-Model
        This works with single step and with batch data
        Args:
            output_lagged (torch.Tensor): Lagged Outputs; shape should be [output_lags] 
            inputs_lagged (list[torch.Tensor]): Lagged Inputs; each shape should be [input_lags[i]]

        Returns:
            torch.Tensor: Output. Shape should be [1]
        """
        # compute input for each subnet
        inputs = self.prepare_inputs(output_lagged, inputs_lagged)
        # evaluate each subnet
        outputs = []
        for i, subnet in enumerate(self.subnets):
            output = subnet(inputs[i])
            outputs.append(output)
        outputs = torch.cat(outputs, dim =1)
        # compute sum of subnet outputs
        return torch.sum(outputs, dim = 1)
    
    def prepare_inputs(self, output_lagged: torch.Tensor, inputs_lagged: list[torch.Tensor]):
        """Compute input for each subnet
        On first glance this function seems more complicated than it needs to be.
        The weird Tensor-Magic is necessary though, to make this compatible with batch-training

        Args:
            output_lagged (torch.Tensor): Lagged Outputs; shape should be [output_lags] 
            inputs_lagged (list[torch.Tensor]): Lagged Inputs; each shape should be [input_lags[i]]

        Returns:
            list[torch.Tensor]: List of inputs for each subnet
        """        
        inputlist = inputs_lagged
        inputlist.append(output_lagged)
        flipped = [tensor.flip(dims = [0]) for tensor in inputlist]
        inputs = []
        for i in range(max([tensor.size(dim=1) for tensor in flipped])):
            input = torch.cat([tensor[:,i].unsqueeze(dim=1) for tensor in flipped if i<tensor.size(dim=1)], dim=1)
            inputs.append(input)
        return inputs

    def identity(x):
        """
        does this need explanation?
        """        
        return x

class LAGNET(nn.Module):
    """Subnet-Modules

    """    
    def __init__(self, n_inputs: int, n_hidden: int, layersize: int , afunc, bias:bool):
        super(LAGNET, self).__init__()
        self.afunc = afunc
        self.bias = bias
        self.n_inputs = n_inputs
        self.linear_layers = [nn.Linear(layersize, layersize, bias = self.bias) for _ in range(n_hidden)]
        self.linear_layers[0] = nn.Linear(self.n_inputs, layersize, bias = self.bias)
        self.linear_layers[-1] = nn.Linear(layersize, 1, bias = self.bias)
        self.linear_layers = nn.ModuleList(self.linear_layers)

    def forward(self, inputs: torch.Tensor):
        """Evaluate a subnet

        Args:
            inputs (torch.Tensor): inputs to the subnet

        Returns:
            torch.Tensor: output of the subnest
        """        
        x = inputs
        for layer in self.linear_layers[:-1]:
            x = self.afunc(layer(x))
        return self.linear_layers[-1](x)
