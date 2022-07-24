import torch
import torch.nn as nn
from .ANARX import ANARX
import numpy as np

class ANARX_SS(nn.Module):
    """Creates a nn.Module that implements the state-space-representation of NN-ANARX-Models from an NN-ANARX-Model
        This is useful, if you need to compute the state-vector, for example in control tasks.
        This model can be exported to ONNX-Format, and used for inference of the state-vector on various target-devices.
    """    
    def __init__(self, reference_model: ANARX):
        """Initializes the SS-representation 

        Args:
            reference_model (ANARX): the ANARX-Model from which to create the SS-representation
        """        
        super(ANARX_SS, self).__init__()
        self.n_subnets = reference_model.n_subnets
        self.subnets = reference_model.subnets
        self.lag_map = reference_model.lag_map
        self.shift_matrix = torch.Tensor(np.eye(self.n_subnets, k=1)).detach()
    
    def forward(self, state: torch.Tensor, y: torch.Tensor, u: torch.Tensor):
        """ Performs one step of prediction
        For closed-loop-predictions y is equal to last_state[0]
        For open-loop-predictions y comes from training data
        For real-time-predictions y comes from the real system

        Args:
            state (torch.Tensor): last state
            y (torch.Tensor): current output
            u (torch.Tensor): current input

        Returns:
            torch.Tensor: next state
        """        
        shifted_state = torch.matmul(self.shift_matrix, state)
        subnetoutputs = torch.zeros(self.n_subnets)
        for i in range(self.n_subnets):
            if self.lag_map[i][0] == 1:
                input = torch.cat((u, y))
            else:
                input = y
            subnetoutputs[i] = self.subnets[i](input)
        next_state = shifted_state + subnetoutputs
        return next_state


