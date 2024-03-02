# ControlNet customization

## Usage: ContolNetFun [OPTIONS]

Process images/videos using Masked ControlNet.

## Options:

- **-input** TEXT: Input image folder (default: ./assets/video)
- **-output** TEXT: Output image folder (default: ./assets/output)
- **-prompt** TEXT: Prompt for image processing (default: humaniod robots in the deep space near nebula)
- **-v, --video**: Enable video mode (off by default)
- **-d, --debug**: Enable debug mode to save intermediate results
- **-l, --limit** INTEGER: Batch size in video mode (default: 2)
- **--help**: Show this message and exit.

## Description:

- **-input:** Specifies the input image folder. Default is "./assets/video".
- **-output:** Specifies the output image folder. Default is "./assets/output".
- **-prompt:** Specifies the prompt for image processing. Default is "humaniod robots in the deep space near nebula".
- **-v, --video:** Enables video mode. By default, this option is disabled.
- **-d, --debug:** Enables debug mode to save intermediate results. By default, this option is disabled.
- **-l, --limit:** Specifies the batch size in video mode. Default is 2. This option is only effective when video mode is enabled.

## Note:

- If video mode is disabled, the program processes images in the input folder individually.
- If video mode is enabled, the program processes images in batches based on the specified limit.

## Example Usage:

To process images with default settings:

```bash
ContolNetFun -input /path/to/input/folder -output /path/to/output/folder -prompt "Custom prompt" -v -d -l 2
```

Run via docker:
```bash
docker build -t cntlnetfun .
docker run --gpus all cntlnetfun
```


## Ideas
- Backround remover  [https://arxiv.org/pdf/2209.09475.pdf]
- Video mode [https://arxiv.org/abs/2303.13439]