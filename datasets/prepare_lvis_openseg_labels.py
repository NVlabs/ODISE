# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import json
import os

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    ann = os.path.join(dataset_dir, "annotations/lvis_v1_val.json")
    print("Loading", ann)
    data = json.load(open(ann, "r"))
    cat_names = [x["name"] for x in sorted(data["categories"], key=lambda x: x["id"])]
    nonrare_names = [
        x["name"]
        for x in sorted(data["categories"], key=lambda x: x["id"])
        if x["frequency"] != "r"
    ]

    synonyms = [x["synonyms"] for x in sorted(data["categories"], key=lambda x: x["id"])]
    nonrare_synonyms = [
        x["synonyms"]
        for x in sorted(data["categories"], key=lambda x: x["id"])
        if x["frequency"] != "r"
    ]

    with open("datasets/openseg/lvis_1203.txt", "w") as f:
        for idx, cat in enumerate(cat_names):
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")

    with open("datasets/openseg/lvis_1203_with_prompt_eng.txt", "w") as f:
        for idx, syns in enumerate(synonyms):
            cat = ",".join(syns)
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")

    with open("datasets/openseg/lvis_nonrare_866.txt", "w") as f:
        for idx, cat in enumerate(nonrare_names):
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")

    with open("datasets/openseg/lvis_nonrare_866_with_prompt_eng.txt", "w") as f:
        for idx, syns in enumerate(nonrare_synonyms):
            cat = ",".join(syns)
            cat = cat.replace("_", " ")
            f.write(f"{idx+1}:{cat}\n")
