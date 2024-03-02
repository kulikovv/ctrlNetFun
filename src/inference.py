import PIL
import cv2
import os
import argparse
from typing import Union, List
import numpy as np
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image, make_image_grid
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from .controlnet import MaskedStableDiffusionControlNetPipeline, MaskedControlNetModel
import torch
from transparent_background import Remover


def build_foreground_masks(input_image: Union[PIL.Image.Image, List[PIL.Image.Image]]) \
        -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
    """
    The building of masks is an separate action from diffusion
    * we use InSPyReNet https://arxiv.org/pdf/2209.09475.pdf
    Can be used instead:
    * MODNet (for people)
    * DINO (for videos)

    Note:
    InSPyReNet takes around 10Gb GPU Memory, so it is good to destroy the Remover once it is not used
    """
    remover = Remover()
    output = []

    images = input_image
    if isinstance(input_image, PIL.Image.Image):
        images = [input_image]

    for img in images:
        output += [remover.process(img, type='map')]

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


def create_pipeline(stable_diffusion_model: str = "runwayml/stable-diffusion-v1-5",
                    controlnet_model: str = "lllyasviel/sd-controlnet-canny")\
        -> MaskedStableDiffusionControlNetPipeline:
    """
    Pipeline initialization is moved to a separate function to avoid code duplicates
    """

    controlnet = MaskedControlNetModel.from_pretrained(
        controlnet_model, torch_dtype=torch.float16, use_safetensors=True)

    pipe = MaskedStableDiffusionControlNetPipeline.from_pretrained(
        stable_diffusion_model, controlnet=controlnet, torch_dtype=torch.float16,
        use_safetensors=True, low_cpu_mem_usage=False
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # to save GPU memory
    pipe.enable_model_cpu_offload()
    return pipe


def image_inference(input_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
                    input_masks: Union[PIL.Image.Image, List[PIL.Image.Image]],
                    input_prompt: Union[str, List[str]]) \
        -> Union[PIL.Image.Image, List[PIL.Image.Image]]:
    """
    Perform T2I inference using a given input image or list of images, masks, and prompts.

    Args:
        input_image (Union[PIL.Image.Image, List[PIL.Image.Image]]):
            Input image or list of input images for inference.
        input_masks (Union[PIL.Image.Image, List[PIL.Image.Image]]):
            Foreground mask or list of foreground masks corresponding to the input images.
        input_prompt (Union[str, List[str]]):
            Input prompt or list of input prompts for the inference process.

    Returns:
        Union[PIL.Image.Image, List[PIL.Image.Image]]:
            Output image or list of output images generated through inference.

    Raises:
        AssertionError: If types of input images and masks don't match or if 
        prompts and images don't have the same number.

    Note:
        - If input_image or input_masks is a single PIL Image, it is converted to a list with one element.
        - If input_prompt is a single string, it is replicated to match the number of images.

    Example:
        >>> input_image = PIL.Image.open('input_image.jpg')
        >>> input_mask = PIL.Image.open('input_mask.jpg')
        >>> input_prompt = "Apply artistic style to the image."
        >>> output_image = image_inference(input_image, input_mask, input_prompt)
    """
    assert type(input_image) is type(
        input_masks), f"Types of input don't match {type(input_image)} and {type(input_masks)}"

    images = input_image
    masks = input_masks
    if isinstance(input_image, PIL.Image.Image):
        images = [input_image]
        masks = [input_masks]
    else:
        assert len(images) == len(masks), "Mask and Images should have same number of images"

    prompts = input_prompt
    if isinstance(input_prompt, str):
        prompts = [input_prompt]*len(images)
    else:
        assert len(images) == len(prompts), "Prompts and Images should have same number"

    pipe = create_pipeline()

    output = []
    for image, mask, prompt in zip(images, masks, prompts):
        output += [pipe(
            prompt,
            image=image,
            mask=mask
        ).images[0]]

    if 1 == len(output):
        return output[0]
    return output


def video_inference(input_image: List[PIL.Image.Image],
                    input_masks: List[PIL.Image.Image],
                    prompt: str = "") \
        -> List[PIL.Image.Image]:
    """
    Perform video inference using a pipeline with provided input images and masks.

    Args:
    - input_image (List[PIL.Image.Image]): List of input images in the form of PIL Image.
    - input_masks (List[PIL.Image.Image]): List of input masks in the form of PIL Image.
    - prompt (str): Text prompt to be used during T2I (default is "").

    Returns:
    - List[PIL.Image.Image]: List of output images resulting from the video inference.

    This function initializes a pipeline for video processing and sets it into video mode.
    It sets attention processors for both the UNet and ControlNet components of the pipeline.
    Prepares the input masks and move them to the appropriate device (GPU if available).
    The provided prompt is duplicated for each input image.
    Finally, the pipeline is invoked with the input prompt, images, and masks to generate output images.
    """

    pipe = create_pipeline()

    # turn this flag on to use the same labels
    pipe.video_mode = True

    # Set the attention processor from https://arxiv.org/pdf/2303.13439.pdf
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

    return pipe(prompt=[prompt] * len(input_image), image=input_image, mask=input_masks).images


def main():
    parser = argparse.ArgumentParser(
        prog='ContolNetFun',
        description='Process images using controlnet')
    parser.add_argument('-input', default="./assets/video")  # input image folder
    parser.add_argument('-output', default="./assets/output")  # out image folder
    parser.add_argument('-prompt', default="humaniod robots in the deep space near nebula")  # propmt
    parser.add_argument('-v', '--video', action='store_true')  # videomode on/off flag
    parser.add_argument('-d', '--debug', action='store_true')  # saves intermediate results if True
    parser.add_argument('-l', '--limit', default=2)  # batchsize in videomode
    args = parser.parse_args()

    # scan the directory for images
    image_path = [os.path.join(args.input, p) for p in os.listdir(
        args.input) if p.endswith(".png") or p.endswith(".jpg")]

    # load images as RGB PIL
    images = [load_image(f) for f in image_path]

    # build foreground masks
    masks = build_foreground_masks(images)

    # generate controlNet guidance images
    canny_images = [create_ctrl_image(im) for im in images]

    # do the inference
    if not args.video:
        rows, cols = 1, 4
        output = image_inference(canny_images, masks, args.prompt)
    else:
        rows, cols = 4, 1
        # we have limit 2, otherwise don't fit 1080Ti GPU
        output = video_inference(canny_images[:args.limit], masks[:args.limit], args.prompt)

    if args.debug:
        # debug mode we store all the images
        for idx, img, mask, canny, out in zip(range(len(images)), images, masks, canny_images, output):
            f = make_image_grid([img, mask, canny, out], rows=rows, cols=cols)
            f.save(f"{args.output}/{idx:03}.png")
    else:
        for i, f in enumerate(output):
            f.save(f"{args.output}/{i:03}.png")


if "__main__" == __name__:
    main()
