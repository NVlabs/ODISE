# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

import logging
import math
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Union
import torch
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as T
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.structures import ImageList
from torch import nn
from torch.nn import functional as F

from ..meta_arch.helper import FeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractorBackbone(Backbone):
    """Backbone implement following for FeatureExtractor

    1. Project same group features into the one single feature map
    2. Sort the features by area, from large to small
    3. Get the stride of each feature map
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        out_features: List[str],
        backbone_in_size: Union[int, Tuple[int]] = (512, 512),
        min_stride: int = 4,
        max_stride: int = 32,
        projection_dim: int = 512,
        num_res_blocks: int = 1,
        use_checkpoint: bool = False,
        slide_training: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.use_checkpoint = use_checkpoint

        self.feature_projections = nn.ModuleList()
        for feature_dim in self.feature_extractor.feature_dims:
            self.feature_projections.append(
                nn.Sequential(
                    *ResNet.make_stage(
                        BottleneckBlock,
                        num_blocks=num_res_blocks,
                        in_channels=feature_dim,
                        bottleneck_channels=projection_dim // 4,
                        out_channels=projection_dim,
                        norm="GN",
                    )
                )
            )

        if isinstance(backbone_in_size, int):
            self.image_preprocess = T.Resize(
                size=backbone_in_size, max_size=1280, interpolation=T.InterpolationMode.BICUBIC
            )
            self.backbone_in_size = (backbone_in_size, backbone_in_size)
            self._slide_inference = False
        else:
            self.image_preprocess = T.Resize(
                size=tuple(backbone_in_size), interpolation=T.InterpolationMode.BICUBIC
            )
            self.backbone_in_size = tuple(backbone_in_size)
            self._slide_inference = True

        self._slide_training = slide_training
        if self._slide_training:
            assert self._slide_inference, "slide training must be used with slide inference"

        self.min_stride = min_stride
        self.max_stride = max_stride

        idx_to_stride = {}
        stride_to_indices = defaultdict(list)
        for indices in self.feature_extractor.grouped_indices:
            for idx in indices:
                stride = self.feature_extractor.feature_strides[idx]
                stride = min(max(stride, self.min_stride), self.max_stride)
                idx_to_stride[idx] = stride
                stride_to_indices[stride].append(idx)

        self._sorted_grouped_indices = []
        for stride in sorted(stride_to_indices.keys()):
            self._sorted_grouped_indices.append(stride_to_indices[stride])

        self._out_feature_channels = {}
        self._out_feature_strides = {}

        for indices in self._sorted_grouped_indices:
            stride = idx_to_stride[indices[0]]
            name = f"s{int(math.log2(stride))}"
            if name not in out_features:
                continue
            assert name not in self._out_feature_strides, f"Duplicate feature name {name}"
            self._out_feature_strides[name] = stride
            self._out_feature_channels[name] = projection_dim
        self._out_features = list(self._out_feature_strides.keys())

        logger.info(
            f"backbone_in_size: {backbone_in_size}, "
            f"slide_training: {self._slide_training}, \n"
            f"slide_inference: {self._slide_inference}, \n"
            f"min_stride: {min_stride}, "
            f"max_stride: {max_stride}, \n"
            f"projection_dim: {projection_dim}, \n"
            f"out_feature_channels: {self._out_feature_channels}\n"
            f"out_feature_strides: {self._out_feature_strides}\n"
            f"use_checkpoint: {use_checkpoint}\n"
        )

    @property
    def size_divisibility(self) -> int:
        return 64

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return destination

    def single_forward(self, img):

        # save memory
        input_image_size = img.shape[-2:]
        # print("input_image_size:", img.shape)
        img = self.image_preprocess(img)
        # print("processed_image_size:", img.shape)
        img = ImageList.from_tensors(list(img), self.size_divisibility).tensor
        # print("padded size:", img.shape)
        features = self.feature_extractor(dict(img=img))

        if self.use_checkpoint:
            return checkpoint.checkpoint(
                self.forward_features, features, input_image_size, use_reentrant=False
            )
        else:
            return self.forward_features(features, input_image_size)

    def forward_features(self, features, input_image_size):
        output_features = {}
        for name, indices in zip(self._out_features, self._sorted_grouped_indices):
            output_feature = None
            stride = self._out_feature_strides[name]
            for idx in indices:
                # print("before restore", name, idx, features[idx].shape, stride)
                # restore aspect ratio
                restored_feature = F.interpolate(
                    features[idx],
                    size=(input_image_size[-2] // stride, input_image_size[-1] // stride),
                )
                projected_feature = self.feature_projections[idx](restored_feature)
                if output_feature is None:
                    output_feature = projected_feature
                else:
                    output_feature = output_feature + projected_feature
            output_features[name] = output_feature

        # for k in output_features:
        #     print(k, output_features[k].shape)

        return output_features

    def slide_forward(self, img):

        batch_size, _, h_img, w_img = img.shape
        # output_features = {k: torch.zeros_like(v) for k, v in self.single_forward(img).items()}
        output_features = {}
        for k in self._out_features:
            stride = self._out_feature_strides[k]
            channel = self._out_feature_channels[k]
            output_features[k] = torch.zeros(
                (batch_size, channel, h_img // stride, w_img // stride),
                dtype=img.dtype,
                device=img.device,
            )

        count_mats = {k: torch.zeros_like(v) for k, v in output_features.items()}

        if self._slide_training:
            short_side = min(min(self.backbone_in_size), min(img.shape[-2:]))
        else:
            # if not slide training then use the shorter side to crop
            short_side = min(img.shape[-2:])

        # h_img, w_img = img.shape[-2:]

        h_crop = w_crop = short_side

        h_stride = w_stride = short_side

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        # print("img.shape:", img.shape)
        # for k in output_features:
        #     print(k, output_features[k].shape)
        # print("h_grids:", h_grids, "w_grids:", w_grids)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                assert crop_img.shape[-2:] == (h_crop, w_crop), f"{crop_img.shape} from {img.shape}"
                # print("crop_img.shape:", crop_img.shape)
                crop_features = self.single_forward(crop_img)
                for k in crop_features:
                    k_x1 = x1 // self._out_feature_strides[k]
                    k_x2 = x2 // self._out_feature_strides[k]
                    k_y1 = y1 // self._out_feature_strides[k]
                    k_y2 = y2 // self._out_feature_strides[k]
                    # output_features[k] += F.pad(
                    #     crop_features[k],
                    #     (
                    #         k_x1,
                    #         output_features[k].shape[-1] - k_x1 - crop_features[k].shape[-1],
                    #         k_y1,
                    #         output_features[k].shape[-2] - k_y1 - crop_features[k].shape[-2],
                    #     ),
                    # )
                    # this version should save some memory
                    output_features[k][:, :, k_y1:k_y2, k_x1:k_x2] += crop_features[k]
                    count_mats[k][..., k_y1:k_y2, k_x1:k_x2] += 1
        assert all((count_mats[k] == 0).sum() == 0 for k in count_mats)

        for k in output_features:
            output_features[k] /= count_mats[k]

        return output_features

    def forward(self, img):
        if (self.training and not self._slide_training) or not self._slide_inference:
            return self.single_forward(img)
        else:
            return self.slide_forward(img)
