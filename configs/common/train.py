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

# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter="???",
    amp=dict(
        enabled=False,
        opt_level=None,
    ),  # options for Automatic Mixed Precision
    grad_clip=None,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000, max_to_keep=2),  # options for PeriodicCheckpointer
    eval_period="${train.checkpointer.period}",
    log_period=50,
    device="cuda",
    seed=42,
    # ...
    wandb=dict(
        enable_writer=False,
        resume=False,
        project="ODISE",
    ),
    cfg_name="",
    run_name="",
    run_tag="",
    reference_world_size=0,
)
