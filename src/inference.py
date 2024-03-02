from diffusers.utils import load_image, make_image_grid
import PIL
import cv2
from typing import Union, List
import numpy as np
from diffusers import UniPCMultistepScheduler
from .controlnet import MaskedStableDiffusionControlNetPipeline, MaskedControlNetModel
import torch
from transparent_background import Remover


def build_foregrond_masks(input_image: Union[PIL.Image.Image, List[PIL.Image.Image]]) \
        -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
    remover = Remover()
    output = []
    if isinstance(input_image, PIL.Image.Image):
        images = [input_image]

    for img in images:
        output += remover.process(img, type='map')

    if 1 == len(output):
        return output[0]
    return output


def create_ctrl_image(pillow_image: PIL.Image.Image,
                      low_threshold: int = 100,
                      high_threshold: int = 200) -> PIL.Image.Image:
    """
    Creates a control net image using Canny edge detection.

    Args:
    - pillow_image (PIL.Image.Image): Input image in the form of a PIL Image.
    - low_threshold (int): Lower threshold value for edge detection (default is 100).
    - high_threshold (int): Higher threshold value for edge detection (default is 200).

    Returns:
    - PIL.Image.Image: Control image with detected edges.

    This function converts the input PIL image into a NumPy array, applies Canny edge
    detection using OpenCV, and then converts the resulting image back to a PIL Image.
    The Canny edge-detected image is replicated across three channels for visualization purposes.
    """
    image = np.array(pillow_image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return PIL.Image.fromarray(image)


def image_inference(input_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
                    input_masks: Union[PIL.Image.Image, List[PIL.Image.Image]]) \
        -> Union[PIL.Image.Image, List[PIL.Image.Image]]:

    assert type(input_image) is type(
        input_masks), f"Types of input don't match {type(input_image)} and {type(input_masks)}"
    controlnet = MaskedControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)

    pipe = MaskedStableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
        use_safetensors=True, low_cpu_mem_usage=False
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    images = input_image
    if isinstance(input_image, PIL.Image.Image):
        images = [input_image]
        masks = [input_masks]

    output = []
    for image, mask in zip(images, masks):
        canny_image = create_ctrl_image(image)

        output += [pipe(
            "the mona lisa in space near nebula shining in the void",
            image=canny_image,
            mask=mask
        ).images[0]]
    if 1 == len(output):
        return output[0]
    return output


def main():
    pass
