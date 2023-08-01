"""Common utility functions for the network"""

import torch


def upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Upsample tensor `source` to have the same spatial size with tensor `target`.

    Parameters
    ----------
    source : torch.Tensor
        The source tensor to be upsampled.
    target : torch.Tensor
        The target tensor to which the source tensor will be upsampled.

    Returns
    -------
    torch.Tensor
        The upsampled `source` tensor.

    Notes
    -----
    This function uses bilinear interpolation to upsample the `source` tensor to
    the spatial size of the `target` tensor. The `source` tensor must have a smaller
    spatial size than the `target` tensor.

    Examples
    --------
    >>> source = torch.rand(1, 3, 256, 256)
    >>> target = torch.rand(1, 3, 512, 512)
    >>> out = upsample_like(source, target)
    >>> assert out.shape == (1, 3, 512, 512)
    """
    source = torch.nn.functional.interpolate(
        source,
        size=target.shape[2:],
        mode="bilinear",
        align_corners=False,
    )
    return source
