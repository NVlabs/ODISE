## Getting Started with ODISE

This document provides a brief intro of the usage of ODISE.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

The [Stable Diffusion v1.3 checkpoint](https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt) and [CLIP checkpoint](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt) will be automatically downloaded to `~/.torch/` and `~/.cache/clip` respectively.
Users should follow the license of the official releases of Stable Diffusion and CLIP.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](README.md#model-zoo),
  for example, `configs/Panoptic/odise_label_coco_50e.py`.
2. We provide `demo/demo.py` that is able to demo builtin configs. Run it with:
```
python demo/demo.py --config-file configs/Panoptic/odise_label_coco_50e.py \
  --input input1.jpg input2.jpg \
  --init-from /path/to/checkpoint_file
  [--other-options]
```
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `train.device=cpu` at end.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Training & Evaluation in Command Line

We provide a script `tools/train_net.py`, that is made to train all the configs provided in ODISE.

To train a model with "train_net.py", first
setup the COCO datasets following
[datasets/README.md](./datasets/README.md#expected-dataset-structure-for-coco),
then run for single node AMP training:
```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 8 --amp 
```
For multi-node AMP training: 
```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 0 --num-machines 2 --dist-url tcp://node_addr:29500 --num-gpus 8 --amp
(node1)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 1 --num-machines 2 --dist-url tcp://node_addr:29500 --num-gpus 8 --amp
```

The configs are made for 16-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
But we provide automatic scaling of learning rate and batch size by passing `--ref $REFERENCE_WORLD_SIZE`. 
For example, if you set `$REFERENCE_WORLD_SIZE=16` and running on 8 GPUs, the batch size and learning rate will be halved (8/16 = 0.5).

```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 8 --amp --ref 16
```

To evaluate a model's performance, run on single node
```
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --num-gpus 8 --eval-only --init-from /path/to/checkpoint
```
or for multi-node inference:
```bash
(node0)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 0 --num-machines 2 --dist-url tcp://node0_addr:29500 --num-gpus 8 --eval-only --init-from /path/to/checkpoint
(node1)$ ./tools/train_net.py --config-file configs/Panoptic/odise_label_coco_50e.py --machine-rank 1 --num-machines 2 --dist-url tcp://node0_addr:29500 --num-gpus 8 --eval-only --init-from /path/to/checkpoint
```

To use the our `odise://` model zoo, you may pass in `--config-file configs/Panoptic/odise_label_coco_50e.py --init-from odise://Panoptic/odise_label_coco_50e` or `--config-file configs/Panoptic/odise_label_coco_50e.py --init-from odise://Panoptic/odise_caption_coco_50e` to `./tools/train_net.py` respectively.
