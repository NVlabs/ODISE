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
