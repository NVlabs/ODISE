## Getting Started with ODISE

This document provides a brief introduction on how to infer with and train ODISE.

For further reading, please refer to [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md).

**Important Note**: ODISE's `demo/demo.py` and `tools/train_net.py` scripts link to the original pre-trained models for [Stable Diffusion v1.3](https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt) and [CLIP](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt). When you run them for the very first time, these scripts will automatically download the pre-trained models for Stable Diffuson and CLIP, from their original sources, to your local directories `$HOME/.torch/` and `$HOME/.cache/clip`, respectively. Their use is subject to the original license terms defined at [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) and [https://github.com/openai/CLIP](https://github.com/openai/CLIP), respectively.


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
[datasets/README.md](./datasets/README.md) and then run, for single-node (8-GPUs) NVIDIA AMP-based training:
```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 8 --amp 
```
For 4-node (32-GPUs) AMP-based training, run: 
```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 0 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --amp
(node1)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 1 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --amp
(node2)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 2 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --amp
(node3)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 3 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --amp
```

Note that our default training configurations are designed for 32 GPUs.
Since we use the AdamW optimizer, it is not clear as to how to scale the learning rate with batch size.
However, we provide the ability to automatically scale the learning rate and the batch size for any number of GPUs used for training by passing in the`--ref $REFERENCE_WORLD_SIZE` argument. 
For example, if you set `$REFERENCE_WORLD_SIZE=32` while training on 8 GPUs, the batch size and learning rate will be set to 8/32 = 0.25 of the original ones.

```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 8 --amp --ref 32
```

ODISE trains in 6 days on 32 NVIDIA V100 GPUs.

To evaluate a trained ODISE model's performance, run on single node
```
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 8 --eval-only --init-from /path/to/checkpoint
```
or for multi-node inference:
```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 0 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --eval-only --init-from /path/to/checkpoint
(node1)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 1 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --eval-only --init-from /path/to/checkpoint
(node2)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 2 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --eval-only --init-from /path/to/checkpoint
(node3)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 3 --num-machines 4 --dist-url tcp://${MASTER_ADDR}:29500 --num-gpus 8 --eval-only --init-from /path/to/checkpoint
```

To use the our provided ODISE [model zoo](README.md#model-zoo), you can pass in the arguments `--config-file configs/Panoptic/odise_label_coco_50e.py --init-from odise://Panoptic/odise_label_coco_50e` or `--config-file configs/Panoptic/odise_label_coco_50e.py --init-from odise://Panoptic/odise_caption_coco_50e` to `./tools/train_net.py`, respectively.
