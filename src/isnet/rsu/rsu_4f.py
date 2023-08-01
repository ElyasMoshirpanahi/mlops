"""Torch module of residual block with 4 downsampling operations and dilated convolution"""

import torch
import torch.nn as nn

from src.isnet.rebnconv.rebnconv import rebnconv


class RSU4F(nn.Module):
    """Implements a modified RSU4 module with dilated convolutions."""

    def __init__(self, in_ch: int = 3, mid_ch: int = 12, out_ch: int = 3) -> None:
        """
        Initialize the class

        Parameters
        ----------
        in_ch : int, optional
            Number of input channels. Default is 3.
        mid_ch : int, optional
            Number of output channels of intermediate convolutional layers. Default is 12.
        out_ch : int, optional
            Number of output channels. Default is 3.
        """
        super().__init__()

        self.rebnconvin = rebnconv(in_ch, out_ch, dirate=1)

        self.rebnconv1 = rebnconv(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = rebnconv(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = rebnconv(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = rebnconv(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = rebnconv(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = rebnconv(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = rebnconv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin
