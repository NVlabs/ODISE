## Getting Started with ODISE

This document provides a brief introduction on how to infer with and train ODISE.

For further reading, please refer to [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md).

**Important Note**: ODISE's `demo/demo.py` and `tools/train_net.py` scripts link to the original pre-trained models for [Stable Diffusion v1.3](https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt) and [CLIP](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt). When you run them for the very first time, these scripts will automatically download the pre-trained models for Stable Diffuson and CLIP, from their original sources, to your local directories `$HOME/.torch/` and `$HOME/.cache/clip`, respectively. Their use is subject to the original license terms defined at [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) and [https://github.com/openai/CLIP](https://github.com/openai/CLIP), respectively.

If you use `stable-diffusion` backbones (latent-diffusion/taming-transformers), initialize optional third_party checkouts first:

```bash
bash tools/bootstrap_third_party.sh
```

If your clone did not include submodules, or if you need a clean refresh:

```bash
bash tools/bootstrap_third_party.sh --force
```
or
```bash
git submodule update --init --recursive
```


### Inference Demo with Pre-trained ODISE Models

1. Choose a model for ODISE and its corresponding configuration file from
  [model zoo](README.md#model-zoo),
  for example, `configs/Panoptic/odise_label_coco_50e.py`. 
  In `demo/demo.py` we also provide a default inbuilt configuration. 
2. Run the `demo/demo.py` with:
```
python demo/demo.py --config-file configs/Panoptic/odise_label_coco_50e.py \
  --input input1.jpg input2.jpg \
  --init-from /path/to/checkpoint_file
  [--other-options]
```
This command will run ODISE's inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo/demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __with a customized vocabulary__, use `--vocab` to specify additional vocabulary names.
* To run __with a caption__, use `--caption` to specify a caption.
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on the cpu__, add `train.device=cpu` at the end.
* To save outputs to a directory (for images) or a file (for webcam or video), use the `--output` option.

The default behavior is to append the user-provided extra vocabulary to the labels from COCO, ADE20K and LVIS.
To use **only** the user-provided vocabulary use `--label ""`.

```
python demo/demo.py --input demo/examples/purse.jpeg --output demo/purse_pred.jpg --label "" --vocab "purse"
```

or

```
python demo/demo.py --input demo/examples/purse.jpeg --output demo/purse_pred.jpg --label "" --caption "there is a black purse"
```

### Command line-based Training & Evaluation

We provide a script `tools/train_net.py` that trains all configurations of ODISE.

To train a model with `tools/train_net.py`, first prepare the datasets following the instructions in
[datasets/README.md](./datasets/README.md) and then run, for CPU-first single-process training:
```bash
./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 1 --force-cpu
```

AMP is only enabled when CUDA is available. On CPU-only machines, training falls back to full precision.

For multi-GPU training (optional, if you still run distributed CUDA), keep your existing launch pattern and pass `--num-gpus` plus `--amp` as before.

### High-throughput Feature Extraction

`tools/extract_features.py` supports distributed extraction. For CPU-only use:

```bash
python tools/extract_features.py \
  --config-file configs/Panoptic/odise_label_coco_50e.py \
  --num-gpus 1 \
  --force-cpu \
  --num-machines 1 \
  --init-from /path/to/checkpoint.pth \
  --output /path/to/feature_out \
  --dataloader dataloader.test \
  --feature-layers s2,s3,s4,s5
``` 

You can scale this to multi-GPU later by increasing `--num-gpus` and `--num-machines` once your environment is configured for distributed execution.

`--dataloader` is a dotted path inside the config; for built-in PANOPTIC configs this is `dataloader.test`.
Each `.pt` file stores a single image's normalized feature maps and metadata and can be merged later as needed.

To evaluate a trained ODISE model on CPU-only single process:
```
./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 1 --force-cpu --eval-only --init-from /path/to/checkpoint
```
or use distributed multi-node/multi-GPU launch flags as needed in your own environment.

To use the our provided ODISE [model zoo](README.md#model-zoo), you can pass in the arguments `--config-file configs/Panoptic/odise_label_coco_50e.py --init-from odise://Panoptic/odise_label_coco_50e` or `--config-file configs/Panoptic/odise_label_coco_50e.py --init-from odise://Panoptic/odise_caption_coco_50e` to `./tools/train_net.py`, respectively.
