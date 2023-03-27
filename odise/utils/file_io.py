# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------
import logging
import os
from detectron2.utils.file_io import PathManager
from iopath.common.file_io import PathHandler


class ODISEHandler(PathHandler):
    """
    Resolve anything that's hosted under ODISE's namespace.
    """

    PREFIX = "odise://"
    URLS = {
        "Panoptic/odise_caption_coco_50e": "https://github.com/NVlabs/ODISE/releases/download/v1.0.0/odise_caption_coco_50e-853cc971.pth",  # noqa
        "Panoptic/odise_label_coco_50e": "https://github.com/NVlabs/ODISE/releases/download/v1.0.0/odise_label_coco_50e-b67d2efc.pth",  # noqa
    }

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    @property
    def local_model_zoo(self):
        return os.environ.get("ODISE_MODEL_ZOO", "")

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        assert name in self.URLS, f"{name} is not a valid ODISE model from {self.URLS.keys()}!"
        path = self.URLS[name]
        if self.local_model_zoo:
            local_path = os.path.join(self.local_model_zoo, os.path.basename(path))
            if os.path.exists(local_path):
                logging.getLogger(__name__).info(f"Using local model zoo: {local_path}.")
                path = local_path
        return PathManager.get_local_path(path, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class StableDiffusionHandler(PathHandler):
    """
    Resolve anything that's hosted under ODISE's namespace.
    """

    PREFIX = "sd://"
    URLS = {
        "v1-3": "https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt",  # noqa
        # following is not used yet
        # we are adding them here for generalization
        "v1-4": "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt",  # noqa
        "v1-5": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt",  # noqa
        "v2-0-base": "https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt",  # noqa
        "v2-0-v": "https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt",  # noqa
        "v2-1-base": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",  # noqa
        "v2-1-v": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt",  # noqa
    }

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    @property
    def local_model_zoo(self):
        return os.environ.get("ODISE_MODEL_ZOO", "")

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        assert name in self.URLS, f"{name} is not a valid ODISE model from {self.URLS.keys()}!"
        path = self.URLS[name]
        if self.local_model_zoo:
            local_path = os.path.join(self.local_model_zoo, os.path.basename(path))
            if os.path.exists(local_path):
                logging.getLogger(__name__).info(f"Using local model zoo: {local_path}.")
                path = local_path
        return PathManager.get_local_path(path, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(ODISEHandler())
PathManager.register_handler(StableDiffusionHandler())
