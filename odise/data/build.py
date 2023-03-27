# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import copy
import logging
import os.path as osp
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils import comm


def get_openseg_labels(dataset, prompt_engineered=False):
    """get the labels in double list format,
    e.g. [[background, bag, bed, ...], ["aeroplane"], ...]
    """

    invalid_name = "invalid_class_id"
    assert dataset in [
        "ade20k_150",
        "ade20k_847",
        "coco_panoptic",
        "pascal_context_59",
        "pascal_context_459",
        "pascal_voc_21",
        "lvis_1203",
    ]

    label_path = osp.join(
        osp.dirname(osp.abspath(__file__)),
        "datasets/openseg_labels",
        f"{dataset}_with_prompt_eng.txt" if prompt_engineered else f"{dataset}.txt",
    )

    # read text in id:name format
    with open(label_path, "r") as f:
        lines = f.read().splitlines()

    categories = []
    for line in lines:
        id, name = line.split(":")
        if name == invalid_name:
            continue
        categories.append({"id": int(id), "name": name})

    return [dic["name"].split(",") for dic in categories]


def prompt_labels(labels, prompt):
    if prompt is None:
        return labels
    labels = copy.deepcopy(labels)
    assert prompt in ["a", "photo", "scene"]
    if prompt == "a":
        for i in range(len(labels)):
            labels[i] = [f"a {l}" for l in labels[i]]
    elif prompt == "photo":
        for i in range(len(labels)):
            labels[i] = [f"a photo of a {l}." for l in labels[i]]
    elif prompt == "scene":
        for i in range(len(labels)):
            labels[i] = [f"a photo of a {l} in the scene." for l in labels[i]]
    else:
        raise NotImplementedError

    return labels


def build_d2_train_dataloader(
    dataset,
    mapper=None,
    total_batch_size=None,
    local_batch_size=None,
    num_workers=0,
    sampler=None,
):

    assert (total_batch_size is None) != (
        local_batch_size is None
    ), "Either total_batch_size or local_batch_size must be specified"

    world_size = comm.get_world_size()

    if total_batch_size is not None:
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size

    if local_batch_size is not None:
        batch_size = local_batch_size

    total_batch_size = batch_size * world_size

    return build_detection_train_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=sampler,
        total_batch_size=total_batch_size,
        aspect_ratio_grouping=True,
        num_workers=num_workers,
        collate_fn=None,
    )


def build_d2_test_dataloader(
    dataset,
    mapper=None,
    total_batch_size=None,
    local_batch_size=None,
    num_workers=0,
):

    assert (total_batch_size is None) != (
        local_batch_size is None
    ), "Either total_batch_size or local_batch_size must be specified"

    world_size = comm.get_world_size()

    if total_batch_size is not None:
        assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
        ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
            total_batch_size, world_size
        )
        batch_size = total_batch_size // world_size

    if local_batch_size is not None:
        batch_size = local_batch_size

    logger = logging.getLogger(__name__)
    if batch_size != 1:
        logger.warning(
            "When testing, batch size is set to 1. "
            "This is the only mode that is supported for d2."
        )

    return build_detection_test_loader(
        dataset=dataset,
        mapper=mapper,
        sampler=None,
        num_workers=num_workers,
        collate_fn=None,
    )
