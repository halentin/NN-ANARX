import torch
import numpy as np

def lag_matrix(tensor, n_lags):
    t_steps = len(tensor)
    y = torch.zeros((t_steps, n_lags))
    new_tensor = torch.cat((torch.ones(n_lags)*tensor[0], tensor))
    for i in range(t_steps):
        y[i,:] = new_tensor[i:i+n_lags]
    return y

def normalize(series):
    offset = np.min(series)
    scale = np.max(series)-np.min(series)
    normalized = (series-offset)/scale
    return normalized, scale, offset
    