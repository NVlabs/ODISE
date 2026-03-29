# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.

import os
import sys


def _bootstrap_vendor_paths() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    third_party = os.path.join(project_root, "third_party")
    for name in ("Mask2Former", "latent-diffusion", "taming-transformers"):
        pkg_root = os.path.join(third_party, name)
        if os.path.isdir(pkg_root) and pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)


_bootstrap_vendor_paths()

__version__ = "0.1"
