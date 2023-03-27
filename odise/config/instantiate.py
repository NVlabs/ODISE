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

from detectron2.config import instantiate


def instantiate_odise(cfg):
    backbone = instantiate(cfg.backbone)
    cfg.sem_seg_head.input_shape = backbone.output_shape()
    cfg.sem_seg_head.pixel_decoder.input_shape = backbone.output_shape()
    cfg.backbone = backbone
    model = instantiate(cfg)

    return model
