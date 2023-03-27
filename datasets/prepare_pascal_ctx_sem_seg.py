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
from pathlib import Path
import shutil

import numpy as np
import tqdm
from PIL import Image
import multiprocessing as mp
import functools
from detail import Detail

# fmt: off
_mapping = np.sort(
    np.array([
        0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22, 23, 397, 25, 284,
        158, 159, 416, 33, 162, 420, 454, 295, 296, 427, 44, 45, 46, 308, 59,
        440, 445, 31, 232, 65, 354, 424, 68, 326, 72, 458, 34, 207, 80, 355,
        85, 347, 220, 349, 360, 98, 187, 104, 105, 366, 189, 368, 113, 115
    ]))
# fmt: on
_key = np.array(range(len(_mapping))).astype("uint8")


def generate_labels(img_info, detail_api, out_dir):
    def _class_to_index(mask, _mapping, _key):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert values[i] in _mapping
        index = np.digitize(mask.ravel(), _mapping, right=True)
        return _key[index].reshape(mask.shape)

    sem_seg = _class_to_index(detail_api.getMask(img_info), _mapping=_mapping, _key=_key)
    sem_seg = sem_seg - 1  # 0 (ignore) becomes 255. others are shifted by 1
    filename = img_info["file_name"]

    Image.fromarray(sem_seg).save(out_dir / filename.replace("jpg", "png"))


def copy_images(img_info, img_dir, out_dir):
    filename = img_info["file_name"]
    shutil.copy2(img_dir / filename, out_dir / filename)


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "pascal_ctx_d2"
    voc_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "VOCdevkit/VOC2010"
    for split in ["training", "validation"]:
        img_dir = voc_dir / "JPEGImages"
        if split == "training":
            detail_api = Detail(voc_dir / "trainval_merged.json", img_dir, "train")
        else:
            detail_api = Detail(voc_dir / "trainval_merged.json", img_dir, "val")
        img_infos = detail_api.getImgs()

        output_img_dir = dataset_dir / "images" / split
        output_ann_dir = dataset_dir / "annotations_ctx59" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir.mkdir(parents=True, exist_ok=True)

        pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

        pool.map(
            functools.partial(copy_images, img_dir=img_dir, out_dir=output_img_dir),
            tqdm.tqdm(img_infos, desc=f"Writing {split} images to {output_img_dir} ..."),
            chunksize=100,
        )

        pool.map(
            functools.partial(generate_labels, detail_api=detail_api, out_dir=output_ann_dir),
            tqdm.tqdm(img_infos, desc=f"Writing {split} images to {output_ann_dir} ..."),
            chunksize=100,
        )
