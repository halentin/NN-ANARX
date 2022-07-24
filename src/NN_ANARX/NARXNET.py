import torch
import torch.nn as nn
from .utilities import lag_matrix

class NARXNET(nn.Module):
    def __init__(self, output_lags: int, input_lags: list[int]):
        """_summary_

        Args:
            output_lags (int): number of output lags
            input_lags (list[int]): number of input lags
        """        
        super(NARXNET, self).__init__()        
        self.output_lags = output_lags
        self.input_lags = input_lags
        self.ninputs = len(input_lags)

        ## State is used for closed loop computation and prediction

    def forward(self, output_lagged: torch.Tensor, inputs_lagged: list[torch.Tensor]):
        """This should be overwritten by the classes that inherit from this

        Args:
            output_lagged (torch.Tensor): Lagged Outputs; shape should be [output_lags] 
            inputs_lagged (list[torch.Tensor]): Lagged Inputs; each shape should be [input_lags[i]]

        Returns:
            torch.Tensor: Output. Shape should be [1]
        """
        return torch.zeros([1])

    def predict(self, u: list[torch.Tensor], state=None):
        """Perform closed loop prediction while calculating gradients.
        This is used for closed loop training

        Args:
            u (list[torch.Tensor]): Inputs to the model
            state (torch.Tensor): initial state
        Returns:
            torch.Tensor: Output
        """ 
        if state == None:
            state = torch.zeros([self.output_lags])
        assert len(u) == self.ninputs
        y_hat = torch.zeros_like(u[0])
        u_lagged = []
        for index, input in enumerate(u):
            u_lagged.append(lag_matrix(input, self.input_lags[index]))
        
        # Predict whole Dataset
        for i in range(len(y_hat)):
            inputs = []
            for input in u_lagged:
                inputs.append(input[i].unsqueeze(dim=0))
            y_hat[i] = self.forward(state.unsqueeze(dim=0), inputs)
            state = torch.roll(state, 1, 0)
            state[0] = y_hat[i]
        return y_hat

    def justpredict(self, u: list[torch.Tensor], state=None):
        """Perform closed loop prediction without calculating gradients.
        This is used for validation and plotting purposes

        Args:
            u (list[torch.Tensor]): Inputs to the model
            state (torch.Tensor): initial state

        Returns:
            torch.Tensor: Output 
        """        
        with torch.no_grad():
            return self.predict(u, state)