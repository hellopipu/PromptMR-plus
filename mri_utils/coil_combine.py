"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from .math import complex_abs_sq, complex_conj, complex_mul
from .fftc import fft2c_new as fft2c
from .fftc import ifft2c_new as ifft2c


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))

def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    """
    Coil Expand with sensitivity maps.
    """
    _, c, _, _, _ = sens_maps.shape
    return fft2c(complex_mul(x.repeat_interleave(c // num_adj_slices, dim=1), sens_maps))

def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    """
    Coil Combine with sensitivity maps.
    """
    b, c, h, w, _ = x.shape
    x = ifft2c(x)
    x = complex_mul(x, complex_conj(sens_maps))
    return x.view(b, num_adj_slices, c // num_adj_slices, h, w, 2).sum(dim=2, keepdim=False)