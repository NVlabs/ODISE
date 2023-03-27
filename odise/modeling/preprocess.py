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

import collections.abc
import torch


def batched_input_to_device(batched_inputs, device, exclude=()):

    if isinstance(exclude, str):
        exclude = [exclude]

    if isinstance(batched_inputs, torch.Tensor):
        batch = batched_inputs.to(device, non_blocking=True)
        return batch
    elif isinstance(batched_inputs, collections.abc.Mapping):
        batch = {}
        for k in batched_inputs:
            if k not in exclude:
                batched_inputs[k] = batched_input_to_device(batched_inputs[k], device)
        return batched_inputs

    elif isinstance(batched_inputs, collections.abc.Sequence) and not isinstance(
        batched_inputs, str
    ):
        return [batched_input_to_device(d, device) for d in batched_inputs]
    elif isinstance(batched_inputs, str):
        return batched_inputs
    else:
        raise TypeError(f"Unsupported type {type(batched_inputs)}")
