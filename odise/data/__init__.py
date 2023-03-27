# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------


from .build import get_openseg_labels, build_d2_train_dataloader, build_d2_test_dataloader
from .dataset_mapper import COCOPanopticDatasetMapper
from .datasets import (
    register_all_ctx59,
    register_all_pascal21,
    register_all_ctx459,
    register_all_coco_panoptic_annos_sem_seg_caption,
)

__all__ = [
    "COCOPanopticDatasetMapper",
    "get_openseg_labels",
    "build_d2_train_dataloader",
    "build_d2_test_dataloader",
    "register_all_ctx59",
    "register_all_pascal21",
    "register_all_ctx459",
    "register_all_coco_panoptic_annos_sem_seg_caption",
]
