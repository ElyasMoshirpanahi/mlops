"""Torch module of residual block"""

import torch
import torch.nn as nn


class rebnconv(nn.Module):
    """
    Residual block consisting of a convolutional layer, batch normalization, and
    ReLU activation.
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dirate: int = 1,
        stride: int = 1,
    ) -> None:
        """
        Initialize the class

        Parameters
        ----------
        in_ch : int, optional
            The number of input channels to the convolutional layer. Default is 3.
        out_ch : int, optional
            The number of output channels from the convolutional layer. Default is 3.
        dirate : int, optional
            The dilation rate of the convolutional layer. Default is 1.
        stride : int, optional
            The stride of the convolutional layer. Default is 1.
        """
        super().__init__()

        self.conv_s1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1 * dirate,
            dilation=1 * dirate,
            stride=stride,
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the residual block given an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the residual block.
        """
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
