import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''
    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1) # C, -> C,1,1
            return scale * x

def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool =  True):
    
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input
    
    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1) # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
        
    return input * noise

class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise. 
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''
    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode
    
    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)