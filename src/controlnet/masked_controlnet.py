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

    # this is an additional flag for prepare_latents function, that should be triggered on to achive latent consistancy
    video_mode = False

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Override of call method, to set the mask if it is provided
        """
        mask = kwds.pop("mask", None)  # get the mask from kwargs
        if isinstance(self.controlnet, MaskedControlNetModel):
            self.controlnet.set_mask(mask)  # set the mask
        else:
            warnings.warn(
                f"To get masked bevaviour, please use MaskedControlNetModel class, got {type(self.controlnet)}")
        # run standard StableDiffusionControlNetPipeline
        return super().__call__(*args, **kwds)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        """
        Override of prepare_latents for video mode generation
        """
        latents = super().prepare_latents(batch_size, num_channels_latents,
                                          height, width, dtype, device, generator, latents)
        if self.video_mode:
            # repeat the same latent for the batch in video mode
            latents = torch.randn((1, 4, latents.shape[2], latents.shape[3]),
                                  device=latents.device,
                                  dtype=latents.dtype).repeat(len(latents), 1, 1, 1)
        return latents
