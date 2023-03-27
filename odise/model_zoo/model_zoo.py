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
from typing import Optional
import pkg_resources
import torch
from detectron2.config import LazyConfig

from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise


class _ModelZooUrls(object):
    """
    Mapping from names to officially released ODISE pre-trained models.
    """

    PREFIX = "https://github.com/NVlabs/ODISE/releases/download/v1.0.0/"

    # format: {config_path.yaml} -> model_{sha256sum}.pth
    CONFIG_PATH_TO_URL_SUFFIX = {
        "Panoptic/odise_caption_coco_50e": "odise_caption_coco_50e-853cc971.pth",
        "Panoptic/odise_label_coco_50e": "odise_label_coco_50e-b67d2efc.pth",
    }

    @staticmethod
    def query(config_path: str) -> Optional[str]:
        """
        Args:
            config_path: relative config filename
        """
        name = config_path.replace(".yaml", "").replace(".py", "")
        if name in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
            suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[name]
            path = os.path.join(_ModelZooUrls.PREFIX, suffix)
            local_model_zoo = os.environ.get("ODISE_MODEL_ZOO", "")
            if local_model_zoo:
                local_path = os.path.join(local_model_zoo, os.path.basename(path))
                if os.path.exists(local_path):
                    logging.getLogger(__name__).info(f"Using local model zoo: {local_path}.")
                    path = local_path
            return path
        return None


def get_checkpoint_url(config_path):
    """
    Returns the URL to the model trained using the given config

    Args:
        config_path (str): config file name relative to ODISE's "configs/"
            directory, e.g., "Panoptic/odise_label_coco_50e.py"

    Returns:
        str: a URL to the model
    """
    url = _ModelZooUrls.query(config_path)
    if url is None:
        raise RuntimeError("Pretrained model for {} is not available!".format(config_path))
    return url


def get_config_file(config_path):
    """
    Returns path to a builtin config file.

    Args:
        config_path (str): config file name relative to ODISE's "configs/"
            directory, e.g., "Panoptic/odise_label_coco_50e.py"

    Returns:
        str: the real path to the config file.
    """
    cfg_file = pkg_resources.resource_filename(
        "odise.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


def get_config(config_path, trained: bool = False):
    """
    Returns a config object for a model in model zoo.

    Args:
        config_path (str): config file name relative to ODISE's "configs/"
            directory, e.g., "Panoptic/odise_label_coco_50e.py"
        trained (bool): If True, will set ``train.init_checkpoint`` to trained model zoo weights.
            If False, the checkpoint specified in the config file's ``train.init_checkpoint``
            is used instead.

    Returns:
        CfgNode or omegaconf.DictConfig: a config object
    """
    cfg_file = get_config_file(config_path)
    assert cfg_file.endswith(".py")
    cfg = LazyConfig.load(cfg_file)
    if trained:
        url = get_checkpoint_url(config_path)
        if "train" in cfg and "init_checkpoint" in cfg.train:
            cfg.train.init_checkpoint = url
        else:
            raise NotImplementedError
    return cfg


def get(config_path, trained: bool = False, device: Optional[str] = None):
    """
    Get a model specified by relative path under ODISE's official ``configs/`` directory.

    Args:
        config_path (str): config file name relative to ODISE's "configs/"
            directory, e.g., "Panoptic/odise_label_coco_50e.py"
        trained (bool): see :func:`get_config`.
        device (str or None): overwrite the device in config, if given.

    Returns:
        nn.Module: a odise model. Will be in training mode.

    Example:
    ::
        from odise import model_zoo
        model = model_zoo.get("Panoptic/odise_label_coco_50e.py", trained=True)
    """
    cfg = get_config(config_path, trained)
    if device is None and not torch.cuda.is_available():
        device = "cpu"
    if device is not None:
        cfg.train.device = device

    model = instantiate_odise(cfg.model)
    if device is not None:
        model = model.to(device)
    if "train" in cfg and "init_checkpoint" in cfg.train:
        ODISECheckpointer(model).load(cfg.train.init_checkpoint)
    return model
