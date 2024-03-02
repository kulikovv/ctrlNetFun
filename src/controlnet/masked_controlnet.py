import warnings
from typing import Any
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from .utils import prepare_mask, apply_mask


class MaskedControlNetModel(ControlNetModel):
    """
    MaskedControlNetModel changes the behaviour of the default ControlNetModel
    """
    mask = None

    def set_mask(self, mask):
        self.mask = mask

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        down_block_res_samples, mid_block_res_sample = super().__call__(*args, **kwds)
        image = args[0]
        assert isinstance(
            image, torch.Tensor), f"the first parameter of controlNet is not a\
                valid tensor {type(image)} check your implementation"
        assert 4 == len(image.shape), f"the input tensor should have 4 dimensions, got {len(image.shape)}"
        # Apply the mask
        if self.mask is not None:
            mask = prepare_mask(mask=self.mask, image=image)

            down_block_res_samples = [apply_mask(
                mask, d) for d in down_block_res_samples]
            mid_block_res_sample = apply_mask(mask, mid_block_res_sample)

        return down_block_res_samples, mid_block_res_sample


class MaskedStableDiffusionControlNetPipeline(StableDiffusionControlNetPipeline):
    """
    We redefine the behaviour of StableDiffusionControlNetPipeline call method, bacause we don't want people who 
    will use MaskedControlNetModel to set manually the mask, what can lead to undesired behaviour and bugs
    """

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        mask = kwds.pop("mask", None)  # get the mask from kwargs
        if isinstance(self.controlnet, MaskedControlNetModel):
            self.controlnet.set_mask(mask)  # set the mask
        else:
            warnings.warn(
                f"To get masked bevaviour, please use MaskedControlNetModel class, got {type(self.controlnet)}")
        # run standard StableDiffusionControlNetPipeline
        return super().__call__(*args, **kwds)
