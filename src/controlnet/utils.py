import numpy as np
import torch
import PIL
from torchvision.transforms import ToTensor
from typing import Union


def prepare_mask(mask: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
                 image: torch.Tensor) -> torch.Tensor:
    """
    Prepares a mask tensor given a mask in the form of PIL Image,
    NumPy array, or PyTorch tensor, and an input image tensor.

    Args:
    - mask: Input mask in the form of PIL Image, NumPy array, or PyTorch tensor.
    - image: Input image tensor.

    Returns:
    - torch.Tensor: Prepared mask tensor suitable controlnet

    Raises:
    - NotImplementedError: If the mask shape or datatype is not supported.
    """
    if isinstance(mask, PIL.Image.Image):
        # convert PIL to torch
        mask = ToTensor()(mask)
        if 3 == len(mask.shape):  # three channels
            mask = mask.unsqueeze(0)
        elif 2 == len(mask.shape):  # single channel
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError(f"mask shape {mask.shape} is not support!")
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
        # we expect mask as input from opencv WxHxC so we will permute
        if 3 == len(mask.shape):
            mask = mask.permute(2, 0, 1).unsqueeze(0)  # WHC -> BCWH
        elif 2 == len(mask.shape):  # single channel
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError(f"mask shape {mask.shape} is not support!")
    elif isinstance(mask, torch.Tensor):
        mask = mask
        if 3 == len(mask.shape):
            mask = mask.unsqueeze(0)
    else:
        raise NotImplementedError(f"mask datatype {type(mask)} is not support!")

    mask = mask[:, 0:1]  # make sure that we will have 1 chanell per mask

    # Ensure the mask has the correct shape and channel dimensions
    assert 4 == len(mask.shape), "Mask is incorrect, please check the documentation"
    assert 1 == mask.shape[1], "Mask should be single channel, please check the documentation"

    mask = mask.to(device=image.device, dtype=image.dtype)

    mask = torch.nn.functional.interpolate(mask, size=image.shape[-2:], mode='bicubic')

    image_batch_size = mask.shape[0]
    repeat_by = image.shape[0] // image_batch_size

    mask = mask.repeat_interleave(repeat_by, dim=0)

    return mask


def apply_mask(
        mask: torch.Tensor,
        tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies the given mask to the input tensor.

    Args:
        mask (torch.Tensor): A tensor representing the mask.
        tensor (torch.Tensor): The tensor to be masked.

    Returns:
        torch.Tensor: The masked tensor.

    Note:
        The function interpolates the mask to match the height and width of the input tensor
        using bicubic interpolation and then applies element-wise multiplication with the
        input tensor to produce the masked tensor.
    """

    height, width = tensor.shape[-2:]
    tensor_mask = torch.nn.functional.interpolate(mask, size=(height, width), mode='bicubic')
    return tensor_mask*tensor
