from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from data.subsample import MaskFunc
from mri_utils import fft2c, ifft2c, rss_complex, complex_abs
from data.subsample import CmrxRecon24MaskFunc, PoissonDiscMaskFunc,CmrxRecon25MaskFunc
def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
    slice_idx: Optional[int] = None,
    num_t: Optional[int] = None,
    num_slc: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    if isinstance(mask_func, CmrxRecon24MaskFunc) or isinstance(mask_func, CmrxRecon25MaskFunc):
        if num_t is not None:
            mask, num_low_frequencies, mask_type = mask_func(shape, offset, seed, slice_idx,num_t,num_slc)
        else:
            mask, num_low_frequencies, mask_type = mask_func(shape, offset, seed)
    else:
        if isinstance(mask_func, PoissonDiscMaskFunc):
            mask_type = 'poisson_disc'
        else:
            mask_type = 'cartesian'
        mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros
    
    if mask.shape[0]!=1: # repeat for coil [cmr24 data]
        mask = mask.repeat_interleave(data.shape[0]//mask.shape[0], dim=0)
    
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies, mask_type


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

class PromptMRSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
        mask_type: The type of mask used.
        num_t: number of temporal frames in the original volume. Only used for CmrxRecon data.
        num_slc: number of slices in the original volume. Only used for CmrxRecon data.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    mask_type: str
    num_t: int
    num_slc: int
    
class CmrxReconDataTransform:
    """
    CmrxRecon23&24 Data Transformer for training
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, uniform_resolution= None, use_seed: bool = True, mask_type: Optional[str] = None, test_num_low_frequencies: Optional[int] = None):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if mask_func is None and mask_type is None:
            raise ValueError("Either `mask_func` or `mask_type` must be specified.")
        if mask_func is not None and mask_type is not None:
            raise ValueError("Both `mask_func` and `mask_type` cannot be set at the same time.")
    
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_resolution = uniform_resolution
        # when training, mask_type will be returned by mask_func
        # when inference, we need to specify mask_type
        # so should check mask_func and mask_type not set at the same time or none at the same time
        if mask_func is None:
            self.mask_type = mask_type
            self.num_low_frequencies = test_num_low_frequencies

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
        num_t: int,
        num_slc: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname)) # so in validation, the same fname (volume) will have the same acc
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1]) 


        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies,mask_type = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end), slice_idx=slice_num, num_t=num_t,num_slc=num_slc
            )
        else:
            masked_kspace = kspace_torch
            mask_torch = to_tensor(mask)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0
            if 'ktRadial' in fname:
                mask_type = 'kt_radial'
            else:
                mask_type = 'cartesian'
            num_low_frequencies = self.num_low_frequencies
            
        sample = PromptMRSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            mask_type=mask_type,
            num_t=num_t,
            num_slc=num_slc
            # attrs=attrs,
        )

        return sample


class FastmriDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, uniform_resolution= None, use_seed: bool = True, mask_type: Optional[str] = None, test_num_low_frequencies: Optional[int] = None):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if mask_func is None and mask_type is None:
            raise ValueError("Either `mask_func` or `mask_type` must be specified.")
        if mask_func is not None and mask_type is not None:
            raise ValueError("Both `mask_func` and `mask_type` cannot be set at the same time.")
    
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_resolution = uniform_resolution
        # when training, mask_type will be returned by mask_func
        # when inference, we need to specify mask_type
        # so should check mask_func and mask_type not set at the same time or none at the same time
        if mask_func is None:
            self.mask_type = mask_type
            self.num_low_frequencies = test_num_low_frequencies
            
    def _crop_if_needed(self, image):
        w_from = h_from = 0
        
        if self.uniform_resolution[0] < image.shape[-3]:
            w_from = (image.shape[-3] - self.uniform_resolution[0]) // 2
            w_to = w_from + self.uniform_resolution[0]
        else:
            w_to = image.shape[-3]
        
        if self.uniform_resolution[1] < image.shape[-2]:
            h_from = (image.shape[-2] - self.uniform_resolution[1]) // 2
            h_to = h_from + self.uniform_resolution[1]
        else:
            h_to = image.shape[-2]

        return image[..., w_from:w_to, h_from:h_to, :]
    
    def _pad_if_needed(self, image):
        pad_w = self.uniform_resolution[0] - image.shape[-3]
        pad_h = self.uniform_resolution[1] - image.shape[-2]
        
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0 
            
        if pad_h > 0:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h - pad_h_left
        else:
            pad_h_left = pad_h_right = 0 
            
        return torch.nn.functional.pad(image.permute(0, 3, 1, 2), (pad_h_left, pad_h_right, pad_w_left, pad_w_right), 'reflect').permute(0, 2, 3, 1)
        
    def _to_uniform_size(self, kspace):
        image = ifft2c(kspace)
        image = self._crop_if_needed(image)
        image = self._pad_if_needed(image)
        kspace = fft2c(image)
        return kspace
    
    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> PromptMRSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """

        is_testing = (target is None)

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)

        # crop to save memory while training
        if self.uniform_resolution is not None:
            if not is_testing:
                kspace_torch = self._to_uniform_size(kspace_torch)
            else:
                # Only crop image height
                if self.uniform_resolution[0] < kspace_torch.shape[-3]:
                    image = ifft2c(kspace_torch)
                    h_from = (image.shape[-3] - self.uniform_resolution[0]) // 2
                    h_to = h_from + self.uniform_resolution[0]
                    image = image[..., h_from:h_to, :, :]
                    kspace_torch = fft2c(image)

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies, mask_type = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )


        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0
            num_low_frequencies = self.num_low_frequencies
            mask_type = self.mask_type

        sample = PromptMRSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            mask_type=mask_type,
            num_t = -1,
            num_slc = -1
        )

        return sample


class CalgaryCampinasDataTransform:
    """
    Data Transformer for training PromptMR models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, uniform_resolution=None, use_seed: bool = True, mask_type: Optional[str] = None, test_num_low_frequencies: Optional[int] = None):
        """
        Args:
            mask_func: A function that can create a mask of
                appropriate shape. Need to specify while training. Defaults to None.
            uniform_resolution: The resolution to which the input data will be cropped.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            mask_type: The type of mask to be used. Need to specify while inference.
            test_num_low_frequencies: The number of low frequencies in ACS. Need to specify while inference for pseudo-radial and poisson-disc masks.
        """
        # when training, mask_type will be returned by mask_func
        # when inference, mask_func is None, we need to specify mask_type
        # so should check mask_func and mask_type not set at the same time or none at the same time
        if mask_func is None and mask_type is None:
            raise ValueError("Either `mask_func` or `mask_type` must be specified.")
        if mask_func is not None and mask_type is not None:
            raise ValueError("Both `mask_func` and `mask_type` cannot be set at the same time.")
    
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_resolution = uniform_resolution

        if mask_func is None: 
            self.mask_type = mask_type
            self.num_low_frequencies = test_num_low_frequencies

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> PromptMRSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname)) # so in validation, the same fname (volume) will have the same acc
        #TODO: cine file does not have left padding
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies, mask_type = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )
        else:
            masked_kspace = kspace_torch
            mask_torch = torch.from_numpy(mask.astype(np.float32))[None,:,:,None]
            num_low_frequencies = self.num_low_frequencies
            mask_type = self.mask_type

        sample = PromptMRSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
            mask_type=mask_type,
            num_t = -1,
            num_slc = -1
        )

        return sample