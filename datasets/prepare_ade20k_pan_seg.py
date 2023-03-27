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

import glob
import json
import os

import numpy as np
import tqdm
from panopticapi.utils import IdGenerator, save_json
from PIL import Image
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

ADE20K_SEM_SEG_CATEGORIES = [c["name"] for c in ADE20K_150_CATEGORIES]
PALETTE = [c["color"] for c in ADE20K_150_CATEGORIES]

if __name__ == "__main__":
    dataset_dir = os.getenv("DETECTRON2_DATASETS", "datasets")

    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(dataset_dir, f"ade/ADEChallengeData2016/images/{dirname}/")
        semantic_dir = os.path.join(dataset_dir, f"ade/ADEChallengeData2016/annotations/{dirname}/")
        instance_dir = os.path.join(
            dataset_dir, f"ade/ADEChallengeData2016/annotations_instance/{dirname}/"
        )

        # folder to store panoptic PNGs
        out_folder = os.path.join(dataset_dir, f"ade/ADEChallengeData2016/ade20k_panoptic_{name}/")
        # json with segmentations information
        out_file = os.path.join(
            dataset_dir, f"ade/ADEChallengeData2016/ade20k_panoptic_{name}.json"
        )

        if not os.path.isdir(out_folder):
            print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
            os.mkdir(out_folder)

        # json config
        config_file = "datasets/ade20k_instance_imgCatIds.json"
        with open(config_file) as f:
            config = json.load(f)

        # load catid mapping
        mapping_file = "datasets/ade20k_instance_catid_mapping.txt"
        with open(mapping_file) as f:
            map_id = {}
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue
                ins_id, sem_id, _ = line.strip().split()
                # shift id by 1 because we want it to start from 0!
                # ignore_label becomes 255
                map_id[int(ins_id) - 1] = int(sem_id) - 1

        ADE20K_150_CATEGORIES = []
        for cat_id, cat_name in enumerate(ADE20K_SEM_SEG_CATEGORIES):
            ADE20K_150_CATEGORIES.append(
                {
                    "name": cat_name,
                    "id": cat_id,
                    "isthing": int(cat_id in map_id.values()),
                    "color": PALETTE[cat_id],
                }
            )
        categories_dict = {cat["id"]: cat for cat in ADE20K_150_CATEGORIES}

        panoptic_json_categories = ADE20K_150_CATEGORIES[:]
        panoptic_json_images = []
        panoptic_json_annotations = []

        filenames = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        for filename in tqdm.tqdm(filenames):
            panoptic_json_image = {}
            panoptic_json_annotation = {}

            image_id = os.path.basename(filename).split(".")[0]

            panoptic_json_image["id"] = image_id
            panoptic_json_image["file_name"] = os.path.basename(filename)

            original_format = np.array(Image.open(filename))
            panoptic_json_image["width"] = original_format.shape[1]
            panoptic_json_image["height"] = original_format.shape[0]

            pan_seg = np.zeros(
                (original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8
            )
            id_generator = IdGenerator(categories_dict)

            filename_semantic = os.path.join(semantic_dir, image_id + ".png")
            filename_instance = os.path.join(instance_dir, image_id + ".png")

            sem_seg = np.asarray(Image.open(filename_semantic))
            ins_seg = np.asarray(Image.open(filename_instance))

            assert sem_seg.dtype == np.uint8
            assert ins_seg.dtype == np.uint8

            semantic_cat_ids = sem_seg - 1
            instance_cat_ids = ins_seg[..., 0] - 1
            # instance id starts from 1!
            # because 0 is reserved as VOID label
            instance_ins_ids = ins_seg[..., 1]

            segm_info = []

            # NOTE: there is some overlap between semantic and instance annotation
            # thus we paste stuffs first

            # process stuffs
            for semantic_cat_id in np.unique(semantic_cat_ids):
                if semantic_cat_id == 255:
                    continue
                if categories_dict[semantic_cat_id]["isthing"]:
                    continue
                mask = semantic_cat_ids == semantic_cat_id
                # should not have any overlap
                assert pan_seg[mask].sum() == 0

                segment_id, color = id_generator.get_id_and_color(semantic_cat_id)
                pan_seg[mask] = color

                area = np.sum(mask)  # segment area computation
                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segm_info.append(
                    {
                        "id": int(segment_id),
                        "category_id": int(semantic_cat_id),
                        "area": int(area),
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )

            # process things
            for thing_id in np.unique(instance_ins_ids):
                if thing_id == 0:
                    continue
                mask = instance_ins_ids == thing_id
                instance_cat_id = np.unique(instance_cat_ids[mask])
                assert len(instance_cat_id) == 1

                semantic_cat_id = map_id[instance_cat_id[0]]

                segment_id, color = id_generator.get_id_and_color(semantic_cat_id)
                pan_seg[mask] = color

                area = np.sum(mask)  # segment area computation
                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segm_info.append(
                    {
                        "id": int(segment_id),
                        "category_id": int(semantic_cat_id),
                        "area": int(area),
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )

            panoptic_json_annotation = {
                "image_id": image_id,
                "file_name": image_id + ".png",
                "segments_info": segm_info,
            }

            Image.fromarray(pan_seg).save(os.path.join(out_folder, image_id + ".png"))

            panoptic_json_images.append(panoptic_json_image)
            panoptic_json_annotations.append(panoptic_json_annotation)

        # save this
        d = {
            "images": panoptic_json_images,
            "annotations": panoptic_json_annotations,
            "categories": panoptic_json_categories,
        }

        save_json(d, out_file)
