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

from copy import deepcopy


def auto_scale_workers(cfg, num_workers: int):
    """
    When the config is defined for certain number of workers (according to
    ``cfg.train.reference_world_size``) that's different from the number of
    workers currently in use, returns a new cfg where the total batch size
    is scaled so that the per-GPU batch size stays the same as the
    original ``total_batch_size // reference_world_size``.

    Other config options are also scaled accordingly:
    * training steps and warmup steps are scaled inverse proportionally.
    * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

    For example, with the original config like the following:

    .. code-block:: yaml

        dataloader.train.total_batch_size: 16
        optimizer.lr: 0.1
        train.reference_world_size: 8
        train.max_iter: 5000
        train.checkpointer.period: 1000

    When this config is used on 16 GPUs instead of the reference number 8,
    calling this method will return a new config with:

    .. code-block:: yaml

        dataloader.train.total_batch_size: 32
        optimizer.lr: 0.2
        train.reference_world_size: 16
        train.max_iter: 2500
        train.checkpointer.period: 500

    Note that both the original config and this new config can be trained on 16 GPUs.
    It's up to user whether to enable this feature (by setting ``reference_world_size``).

    Returns:
        CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
    """
    old_world_size = cfg.train.reference_world_size
    if old_world_size == 0 or old_world_size == num_workers:
        print("No need to scale the config.")
        return cfg
    cfg = deepcopy(cfg)

    assert cfg.dataloader.train.total_batch_size % old_world_size == 0, (
        f"Invalid reference_world_size in config! "
        f"{cfg.dataloader.train.total_batch_size} % {old_world_size} != 0"
    )
    scale = num_workers / old_world_size
    bs = cfg.dataloader.train.total_batch_size = int(
        round(cfg.dataloader.train.total_batch_size * scale)
    )
    lr = cfg.optimizer.lr = cfg.optimizer.lr * scale
    max_iter = cfg.train.max_iter = int(round(cfg.train.max_iter / scale))
    cfg.train.eval_period = int(round(cfg.train.eval_period / scale))
    cfg.train.checkpointer.period = int(round(cfg.train.checkpointer.period / scale))
    cfg.train.reference_world_size = num_workers  # maintain invariant
    print(
        f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, " f"max_iter={max_iter}."
    )

    return cfg
