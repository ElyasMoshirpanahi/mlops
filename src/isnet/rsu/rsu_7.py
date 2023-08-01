"""Torch module of residual block with 7 downsampling operations"""

import torch
import torch.nn as nn

from src.isnet.network_utils import upsample_like
from src.isnet.rebnconv.rebnconv import rebnconv


class RSU7(nn.Module):
    """Residual U-Net Block with 7 downsampling operations."""

    def __init__(
        self,
        in_ch: int = 3,
        mid_ch: int = 12,
        out_ch: int = 3,
        img_size: int = 512,
    ) -> None:
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
        img_size : int, optional
            The size of the image. Default is 512.
        """
        super().__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = rebnconv(in_ch, out_ch, dirate=1)  # 1 -> 1/2

        self.rebnconv1 = rebnconv(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = rebnconv(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = rebnconv(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = rebnconv(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = rebnconv(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = rebnconv(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = rebnconv(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = rebnconv(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = rebnconv(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin
