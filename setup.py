#!/usr/bin/env python
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
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "odise", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    odise/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(path.dirname(path.realpath(__file__)), "odise", "model_zoo", "configs")
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "configs/**/*.py", recursive=True
    )
    return config_paths


setup(
    name="odise",
    version=get_version(),
    author="Jiarui Xu",
    url="https://github.com/NVlabs/ODISE",
    description="Open-vocabulary DIffusion-based Panoptic Segmentation",
    packages=find_packages(exclude=("configs", "tests*")),
    package_data={"odise.model_zoo": get_model_zoo_configs()},
    python_requires=">=3.8",
    install_requires=[
        "timm==0.6.11",  # freeze timm version for stabliity
        "opencv-python==4.6.0.66",
        "diffdist==0.1",
        "nltk>=3.6.2",
        "einops>=0.3.0",
        "wandb>=0.12.11",
        # "transformers==4.20.1",  # freeze transformers version for stabliity
        # there is BC breaking in omegaconf 2.2.1
        # see: https://github.com/omry/omegaconf/issues/939
        "omegaconf==2.1.1",
        "open-clip-torch==2.0.2",
        f"mask2former @ file://localhost/{os.getcwd()}/third_party/Mask2Former/",
        "stable-diffusion-sdkit==2.1.3",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install 'odise[dev]'`
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "flake8-bugbear",
            "flake8-comprehensions",
            "click==8.0.4",
            "importlib-metadata==4.11.3",
        ],
    },
    include_package_data=True,
)
