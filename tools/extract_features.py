#!/usr/bin/env python
#
# ------------------------------------------------------------------------------
# Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This work is made available under the Nvidia Source Code License.
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import sys
from contextlib import nullcontext
from typing import Dict, List, Optional

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MASK2FORMER_PATH = osp.join(PROJECT_ROOT, "third_party", "Mask2Former")
if osp.isdir(MASK2FORMER_PATH) and MASK2FORMER_PATH not in sys.path:
    sys.path.insert(0, MASK2FORMER_PATH)
LATENT_DIFFUSION_PATH = osp.join(PROJECT_ROOT, "third_party", "latent-diffusion")
if osp.isdir(LATENT_DIFFUSION_PATH) and LATENT_DIFFUSION_PATH not in sys.path:
    sys.path.insert(0, LATENT_DIFFUSION_PATH)
TAMING_TRANSFORMERS_PATH = osp.join(PROJECT_ROOT, "third_party", "taming-transformers")
if osp.isdir(TAMING_TRANSFORMERS_PATH) and TAMING_TRANSFORMERS_PATH not in sys.path:
    sys.path.insert(0, TAMING_TRANSFORMERS_PATH)

import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import create_ddp_model, default_argument_parser, launch
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from odise.checkpoint import ODISECheckpointer
from odise.config import auto_scale_workers, instantiate_odise
from odise.engine.defaults import default_setup, get_model_from_module


def _resolve_cfg_entry(cfg, dotted_key: str):
    target = cfg
    for part in dotted_key.split("."):
        if not hasattr(target, part):
            raise ValueError(f"Cannot find config entry '{dotted_key}' at '{part}'.")
        target = getattr(target, part)
    return target


def _safe_image_id(sample: Dict, fallback: int) -> str:
    image_id = sample.get("image_id")
    if image_id is None:
        image_id = sample.get("id")
    if image_id is None:
        image_id = sample.get("file_name", f"sample_{fallback}")
    return str(image_id).replace("/", "_")


def _assert_file_exists(path: str, label: str) -> None:
    if not path:
        raise ValueError(f"{label} is required and cannot be empty.")
    if path.startswith(("odise://", "http://", "https://")):
        return
    if not osp.exists(path):
        raise ValueError(f"{label} does not exist: {path}")


def _filter_layers(features: Dict[str, torch.Tensor], layer_names: Optional[List[str]]) -> Dict[str, torch.Tensor]:
    if not layer_names:
        return features

    missing = [name for name in layer_names if name not in features]
    if missing:
        raise KeyError(f"Requested feature layers not present: {missing}")
    return {name: features[name] for name in layer_names}


def _get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        return p.device
    raise ValueError("Could not infer model device: model has no parameters and no device attribute.")


@torch.no_grad()
def extract_features(cfg, args):
    cfg = auto_scale_workers(cfg, comm.get_world_size())
    if args.init_from:
        cfg.train.init_checkpoint = args.init_from
    if args.output:
        cfg.train.output_dir = args.output
    cfg.train.log_dir = cfg.train.output_dir
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    default_setup(cfg, args)
    logger = setup_logger(cfg.train.log_dir, distributed_rank=comm.get_rank(), name="odise")

    logger.info(f"Running with config:\n{LazyConfig.to_py(cfg)}")
    logger.info(
        f"extract_features args: num_gpus={args.num_gpus}, num_machines={args.num_machines}, "
        f"dataloader={args.dataloader}, feature_layers={args.feature_layers or 'ALL'}, "
        f"output={args.output}, output_dtype={args.output_dtype}, max_images={args.max_images}"
    )

    model = instantiate_odise(cfg.model)
    if getattr(args, "force_cpu", False) and cfg.train.device == "cuda":
        logger.warning("CPU-only execution requested via --force-cpu. Setting cfg.train.device=cpu.")
        cfg.train.device = "cpu"
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    model_module = get_model_from_module(model)
    model_device = _get_model_device(model_module)

    if cfg.train.init_checkpoint:
        _assert_file_exists(cfg.train.init_checkpoint, "Checkpoint path")
    checkpointer = ODISECheckpointer(model, cfg.train.output_dir)
    if cfg.train.init_checkpoint:
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    else:
        raise ValueError("`--init-from` is required for extraction.")

    model.eval()

    dataloader_cfg = _resolve_cfg_entry(cfg, args.dataloader)
    data_loader = instantiate(dataloader_cfg)

    if getattr(args, "force_cpu", False) and cfg.train.device != "cpu":
        logger.warning("CPU-only execution requested via --force-cpu. Forcing feature extraction to CPU.")
        cfg.train.device = "cpu"
    elif cfg.train.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, switching feature extraction to CPU.")
        cfg.train.device = "cpu"
    if args.amp and not torch.cuda.is_available():
        logger.warning("AMP requested but CUDA is unavailable; running without autocast.")
    amp_ctx = torch.amp.autocast(
        "cuda", enabled=args.amp and torch.cuda.is_available() and not getattr(args, "force_cpu", False)
    ) if torch.cuda.is_available() and not getattr(args, "force_cpu", False) else nullcontext()

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    layer_names = [name.strip() for name in args.feature_layers.split(",") if name.strip()]
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    output_dtype = dtype_map[args.output_dtype]
    output_root = osp.join(cfg.train.output_dir, "features")
    rank_root = osp.join(output_root, f"rank_{rank:02d}_of_{world_size:02d}")
    PathManager.mkdirs(rank_root)
    logger.info(f"Writing feature shards to {rank_root}")

    processed = 0
    for batch_idx, batched_inputs in enumerate(data_loader):
        if args.max_images > 0 and processed >= args.max_images:
            break
        images = [sample["image"].to(device=model_device, non_blocking=True) for sample in batched_inputs]
        images = [(x - model_module.pixel_mean) / model_module.pixel_std for x in images]
        image_batch = ImageList.from_tensors(images, model_module.size_divisibility)

        with amp_ctx:
            features = model_module.backbone(image_batch.tensor)

        features = _filter_layers(features, layer_names)

        for local_idx, sample in enumerate(batched_inputs):
            if args.max_images > 0 and processed >= args.max_images:
                break
            feature_entry = {}
            for name, value in features.items():
                feature_entry[name] = value[local_idx].to(dtype=output_dtype).cpu()

            image_id = _safe_image_id(sample, batch_idx * len(batched_inputs) + local_idx)
            payload = {
                "image_id": sample.get("image_id", image_id),
                "file_name": sample.get("file_name"),
                "height": sample.get("height"),
                "width": sample.get("width"),
                "layer_names": sorted(feature_entry.keys()),
                "features": feature_entry,
            }
            out_file = osp.join(
                rank_root,
                f"{image_id}_bs{local_idx:02d}_r{rank:02d}.pt",
            )
            if args.skip_existing and PathManager.exists(out_file):
                processed += 1
                continue
            torch.save(payload, out_file)
            processed += 1

        if processed % 50 == 0 and comm.is_main_process():
            logger.info(f"Rank {rank}: processed {processed} samples")

    comm.synchronize()
    if comm.is_main_process():
        logger.info(f"Feature extraction finished with total_local={processed}.")


def parse_args():
    parser = default_argument_parser()
    parser.add_argument("--output", required=True, type=str, help="Output directory for feature shards")
    parser.add_argument(
        "--dataloader",
        default="dataloader.test",
        type=str,
        help="Config key path for dataloader, for example `dataloader.test`.",
    )
    parser.add_argument(
        "--feature-layers",
        default="",
        type=str,
        help="Comma-separated backbone feature keys. Leave empty to export all.",
    )
    parser.add_argument(
        "--output-dtype",
        default="fp16",
        type=str,
        choices=["fp16", "fp32", "bf16"],
        help="Dtype to store extracted feature tensors.",
    )
    parser.add_argument("--max-images", default=-1, type=int, help="Stop after N images per rank.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples already written.")
    parser.add_argument("--amp", action="store_true", help="Use AMP for backbone inference.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU-only execution")
    parser.add_argument("--init-from", type=str, default="", help="Model checkpoint path.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.force_cpu or not torch.cuda.is_available():
        if args.num_gpus != 1:
            print("CPU-only execution requested. Forcing --num-gpus=1 for feature extraction.")
        args.num_gpus = 1
    if args.force_cpu or (args.amp and not torch.cuda.is_available()):
        if args.amp and not torch.cuda.is_available():
            print("GPU-only AMP requested without CUDA. Forcing --amp disabled for feature extraction.")
        if args.force_cpu and args.num_gpus != 1:
            print("CPU-only execution requested. Forcing --num-gpus=1 for feature extraction.")
        args.amp = False
    cfg = LazyConfig.load(args.config_file)
    launch(
        extract_features,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(cfg, args),
    )
