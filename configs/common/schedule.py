# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/detectron2/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from fvcore.common.param_scheduler import CosineParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

cosine_lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=0.01),
    warmup_length="???",
    warmup_method="linear",
    warmup_factor=0.001,
)
