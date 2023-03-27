# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

# Convert adding COCO captions into annotation json

import json
import os
from collections import defaultdict


def load_coco_caption():
    id2caption = defaultdict(list)
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    for json_file in ["captions_train2017.json", "captions_val2017.json"]:
        with open(os.path.join(dataset_dir, "annotations", json_file)) as f:
            obj = json.load(f)
            for ann in obj["annotations"]:
                id2caption[int(ann["image_id"])].append(ann["caption"])

    return id2caption


def create_annotation_with_caption(input_json, output_json):
    id2coco_caption = load_coco_caption()

    with open(input_json) as f:
        obj = json.load(f)

    coco_count = 0

    print(f"Starting to add captions to {input_json} ...")
    print(f"Total images: {len(obj['annotations'])}")
    for ann in obj["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in id2coco_caption:
            ann["coco_captions"] = id2coco_caption[image_id]
            coco_count += 1
    print(f"Found {coco_count} captions from COCO ")

    print(f"Start writing to {output_json} ...")
    with open(output_json, "w") as f:
        json.dump(obj, f)


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    for s in ["val2017", "val2017_100", "train2017"]:
        create_annotation_with_caption(
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "annotations/panoptic_caption_{}.json".format(s)),
        )
