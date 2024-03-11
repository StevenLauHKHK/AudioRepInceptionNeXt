import torch
import torch.nn as nn
from torch import Tensor


class PConv(nn.Module):
    """ Partial convolution (PConv).
    """
    def __init__(self,
                dim: int,
                n_div: int,
                forward: str = "split_cat",
                kernel_size: int = [3, 3],
                stride: int  = [1, 1],
                padding: int = [1, 1],
                bias: bool = False) -> None:
    
        """ Construct a PConv layer.
        :param dim: Number of input/output channels
        :param n_div: Reciprocal of the partial ratio.
        :param forward: Forward type, can be either ’split_cat’ or ’slicing’.
        :param kernel_size: Kernel size.
        """

        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    
    def forward_slicing(self, x: Tensor) -> Tensor:
        """ Apply forward pass for inference. """
        x = x.clone()
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x
    
    def forward_split_cat(self, x: Tensor) -> Tensor:
        """ Apply forward pass for training. """
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), 1)

        return x