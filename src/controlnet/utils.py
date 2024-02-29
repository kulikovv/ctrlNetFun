import numpy as np
import torch
import PIL
from torchvision.transforms import ToTensor


def prepare_mask(
        mask,  # Input mask (can be PIL Image, numpy array, or torch Tensor)
        width,  # Width of the target mask
        height,  # Height of the target mask
        batch_size,  # Batch size for the mask
        num_images_per_prompt,  # Number of images per prompt
        device,  # Device to move the mask tensor
        dtype,  # Data type of the mask tensor
        do_classifier_free_guidance=False,  # Flag indicating whether to do classifier-free guidance
        guess_mode=False,  # Flag indicating guess mode
):
    """
    Prepare the mask tensor for the given parameters.

    Args:
        mask: Input mask, can be a PIL Image, numpy array, or torch Tensor.
        width: Width of the target mask.
        height: Height of the target mask.
        batch_size: Batch size for the mask.
        num_images_per_prompt   : Number of images per prompt.
        device: Device to move the mask tensor.
        dtype: Data type of the mask tensor.
        do_classifier_free_guidance (bool): Flag indicating whether to do classifier-free guidance.
        guess_mode (bool): Flag indicating guess mode.

    Returns:
        torch.Tensor: Prepared mask tensor.

    Raises:
        NotImplementedError: If the datatype of the mask is not supported.
        AssertionError: If the mask is incorrect according to specifications.

    Note:
        The function handles different types of mask inputs (PIL Image, numpy array, or torch Tensor),
        interpolates them to match the target size, repeats them according to the batch size,
        and performs necessary transformations based on additional flags provided.
    """

    if isinstance(mask, PIL.Image.Image):
        # convert PIL to torch
        mask = ToTensor()(mask)
        if 3 == len(mask.shape):
            mask = mask.unsqueeze(0)
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
        # we expect mask as input from opencv WxHxC so we will permute
        if 3 == len(mask.shape):
            mask = mask.permute(2, 0, 1).unsqueeze(0)
    elif isinstance(mask, torch.Tensor):
        mask = mask
        if 3 == len(mask.shape):
            mask = mask.unsqueeze(0)
    else:
        raise NotImplementedError(f"mask datatype {type(mask)} is not support!")

    # Ensure the mask has the correct shape and channel dimensions
    assert 4 == len(mask.shape), "Mask is incorrect, please check the documentation"
    assert 1 == mask.shape[1], "Mask should be single channel, please check the documentation"

    mask = torch.nn.functional.interpolate(mask, size=(height, width), mode='bicubic')

    image_batch_size = mask.shape[0]
    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # mask batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    mask = mask.repeat_interleave(repeat_by, dim=0)

    mask = mask.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        mask = torch.cat([mask] * 2)

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
