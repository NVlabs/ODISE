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

import os
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.io as sio

import tqdm


def generate_labels(mat_file, out_dir):

    mat = sio.loadmat(mat_file)
    label_map = mat["LabelMap"]
    assert label_map.dtype == np.uint16
    label_map[label_map == 0] = 65535
    label_map = label_map - 1
    label_map[label_map == 65534] = 65535

    out_file = out_dir / Path(mat_file.name).with_suffix(".tif")
    Image.fromarray(label_map).save(out_file)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "pascal_ctx_d2"
    voc_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit/VOC2010"
    mat_dir = voc_dir / "trainval"
    for split in ["training", "validation"]:
        file_names = list((dataset_dir / "images" / split).glob("*.jpg"))
        output_img_dir = dataset_dir / "images" / split
        output_ann_dir = dataset_dir / "annotations_ctx459" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

        for file_name in tqdm.tqdm(file_names):
            mat_file_path = mat_dir / f"{file_name.stem}.mat"

            generate_labels(mat_file_path, output_ann_dir)
