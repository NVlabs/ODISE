# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# To view a copy of this license, visit
# https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
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
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import tqdm
from PIL import Image
from mask2former.data.datasets.register_ade20k_full import ADE20K_SEM_SEG_FULL_CATEGORIES


def loadAde20K(file):
    fileseg = file.replace(".jpg", "_seg.png")
    with Image.open(fileseg) as io:
        seg = np.array(io)

    R = seg[:, :, 0]
    G = seg[:, :, 1]
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

    return {"img_name": file, "segm_name": fileseg, "class_mask": ObjectClassMasks}


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    index_file = dataset_dir / "ade/ADE20K_2021_17_01" / "index_ade20k.pkl"
    with open(index_file, "rb") as f:
        index_ade20k = pkl.load(f)

    id_map = {}
    for cat in ADE20K_SEM_SEG_FULL_CATEGORIES:
        id_map[cat["id"]] = cat["trainId"]

    # make output dir
    for name in ["training", "validation"]:
        image_dir = dataset_dir / "ade/ADE20K_2021_17_01" / "images_detectron2" / name
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir = dataset_dir / "ade/ADE20K_2021_17_01" / "annotations_detectron2" / name
        annotation_dir.mkdir(parents=True, exist_ok=True)

    # process image and gt
    for folder_name, file_name in tqdm.tqdm(
        zip(index_ade20k["folder"], index_ade20k["filename"]),
        total=len(index_ade20k["filename"]),
    ):
        split = "validation" if file_name.split("_")[1] == "val" else "training"
        info = loadAde20K(str(dataset_dir / "ade" / folder_name / file_name))

        # resize image and label
        img = np.asarray(Image.open(info["img_name"]))
        lab = np.asarray(info["class_mask"])

        h, w = img.shape[0], img.shape[1]
        max_size = 512
        resize = True
        if w >= h > max_size:
            h_new, w_new = max_size, round(w / float(h) * max_size)
        elif h >= w > max_size:
            h_new, w_new = round(h / float(w) * max_size), max_size
        else:
            resize = False

        if resize:
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

        assert img.dtype == np.uint8
        assert lab.dtype == np.int32

        # apply label conversion and save into uint16 images
        output = np.zeros_like(lab, dtype=np.uint16) + 65535
        for obj_id in np.unique(lab):
            if obj_id in id_map:
                output[lab == obj_id] = id_map[obj_id]

        output_img = dataset_dir / "ade/ADE20K_2021_17_01" / "images_detectron2" / split / file_name
        output_lab = (
            dataset_dir
            / "ade/ADE20K_2021_17_01"
            / "annotations_detectron2"
            / split
            / file_name.replace(".jpg", ".tif")
        )
        Image.fromarray(img).save(output_img)

        assert output.dtype == np.uint16
        Image.fromarray(output).save(output_lab)
