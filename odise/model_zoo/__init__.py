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

"""
Model Zoo API for ODISE: a collection of functions to create common model architectures
listed in `MODEL_ZOO.md <https://github.com/NVlabs/ODISE/blob/master/README.md#model-zoo>`_,
and optionally load their pre-trained weights.
"""

from .model_zoo import get, get_config_file, get_checkpoint_url, get_config

__all__ = ["get_checkpoint_url", "get", "get_config_file", "get_config"]
