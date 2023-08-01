"""
ISNetDIS architecture class

Notes
-----
This implementation uses residual U-Net blocks with max pooling layers for
the encoder, and transposed convolutions for the decoder. The side output
layers produce segmentation maps with the same spatial resolution as the
input.

Examples
--------
>>> net = ISNetDIS()
>>> x = torch.rand(1, 3, 256, 256)
>>> out = net(x)
>>> assert out.shape == (1, 1, 256, 256)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from src.isnet.network_utils import upsample_like
from src.isnet.rsu import rsu_4, rsu_4f, rsu_5, rsu_6, rsu_7


class ISNetDIS(nn.Module):
    """Implements the ISNetDIS architecture."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        """
        Initializes the class

        Parameters
        ----------
        in_ch : int, optional
            Number of input channels. Default is 3.
        out_ch : int, optional
            Number of output channels. Default is 1.
        """
        super().__init__()

        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = rsu_7.RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = rsu_6.RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = rsu_5.RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = rsu_4.RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = rsu_4f.RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = rsu_4f.RSU4F(512, 256, 512)

        # decoder
        self.stage5d = rsu_4f.RSU4F(1024, 256, 512)
        self.stage4d = rsu_4.RSU4(1024, 128, 256)
        self.stage3d = rsu_5.RSU5(512, 64, 128)
        self.stage2d = rsu_6.RSU6(256, 32, 64)
        self.stage1d = rsu_7.RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of the ISNetDIS architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        tuple of lists of torch.Tensor
            A tuple containing two lists of tensors:
            - the first list contains six side outputs of shape:
              (batch_size, out_ch, height, width) for each stage of the network,
              after being passed through a sigmoid function.
            - the second list contains the intermediate feature maps at each stage of
              the encoder, in decreasing order of size, i.e. hx1d to hx6 of shapes
              [(batch_size, 64, h/64, w/64), ..., (batch_size, 512, h/2, w/2)].
        """
        hx = x
        hxin = self.conv_in(hx)

        # encoder
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = upsample_like(d6, x)

        return [
            torch.sigmoid(d1),
            torch.sigmoid(d2),
            torch.sigmoid(d3),
            torch.sigmoid(d4),
            torch.sigmoid(d5),
            torch.sigmoid(d6),
        ], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]
