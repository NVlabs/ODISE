#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
from os import path
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "mask2former", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


# Copied from Detectron2
def get_extensions():
    # skip building
    if not (os.environ.get("FORCE_CUDA") or torch.cuda.is_available()) or CUDA_HOME is None:
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "mask2former/modeling/pixel_decoder/ops/src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Force cuda since torch ask for a device, not if cuda is in fact available.
    if (os.environ.get("FORCE_CUDA") or torch.cuda.is_available()) and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        if CUDA_HOME is None:
            raise NotImplementedError(
                "CUDA_HOME is None. Please set environment variable CUDA_HOME."
            )
        else:
            raise NotImplementedError(
                "No CUDA runtime is found. Please set FORCE_CUDA=1 or test it by running torch.cuda.is_available()."  # noqa
            )

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="mask2former",
    version=get_version(),
    author="Bowen Cheng", # Thanks Bowen! 
    url="https://github.com/facebook/mask2former",
    description="A pip installable version of mask2former",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.6",
    install_requires=[
        "detectron2 @ https://github.com/facebookresearch/detectron2/archive/v0.6.zip",
        "scipy>=1.7.3",
        "boto3>=1.21.25",
        "hydra-core==1.1.1",
        # there is BC breaking in omegaconf 2.2.1
        # see: https://github.com/omry/omegaconf/issues/939
        "omegaconf==2.1.1",
        "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
        "lvis @ https://github.com/lvis-dataset/lvis-api/archive/master.zip",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
