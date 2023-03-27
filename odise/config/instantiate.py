# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
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
