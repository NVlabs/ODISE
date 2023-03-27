# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import os
from detectron2.data import MetadataCatalog
from mask2former.data.datasets.register_coco_panoptic_annos_semseg import (
    get_metadata,
    register_coco_panoptic_annos_sem_seg,
)

_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {
    "coco_2017_train_panoptic_caption": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_caption_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_semseg_train2017",
    ),
    "coco_2017_val_panoptic_caption": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_caption_val2017.json",
        "coco/panoptic_semseg_val2017",
    ),
    "coco_2017_val_100_panoptic_caption": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_caption_val2017_100.json",
        "coco/panoptic_semseg_val2017_100",
    ),
}


# NOTE: the name is "coco_2017_train_panoptic_caption_with_sem_seg" and "coco_2017_val_panoptic_caption_with_sem_seg" # noqa
def register_all_coco_panoptic_annos_sem_seg_caption(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        if prefix.endswith("_panoptic_caption"):
            prefix_instances = prefix[: -len("_panoptic_caption")]
        else:
            raise ValueError("Unknown prefix: {}".format(prefix))
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


register_all_coco_panoptic_annos_sem_seg_caption(os.getenv("DETECTRON2_DATASETS", "datasets"))
