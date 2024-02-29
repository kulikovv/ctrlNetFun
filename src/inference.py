from diffusers.utils import load_image, make_image_grid
import PIL
import cv2
import numpy as np
from diffusers import ControlNetModel, UniPCMultistepScheduler
from .controlnet import MaskedStableDiffusionControlNetPipeline
import torch


def create_ctrl_image(pillow_image: PIL.Image.Image,
                      low_threshold: int = 100,
                      high_threshold: int = 200) -> PIL.Image.Image:
    image = np.array(pillow_image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


def main():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)

    pipe = MaskedStableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
        use_safetensors=True, low_cpu_mem_usage=False
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
