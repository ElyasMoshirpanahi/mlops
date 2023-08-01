"""Torch module of residual block with 4 downsampling operations"""

import torch
import torch.nn as nn

from src.isnet.network_utils import upsample_like
from src.isnet.rebnconv.rebnconv import rebnconv


class RSU4(nn.Module):
    """Residual U-Net Block with 4 downsampling operations."""

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
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = rebnconv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = rebnconv(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = rebnconv(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = rebnconv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin
