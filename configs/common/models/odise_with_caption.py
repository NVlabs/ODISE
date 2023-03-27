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

from detectron2.config import LazyCall as L
from odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from odise.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from .mask_generator_with_caption import model

model.backbone = L(FeatureExtractorBackbone)(
    feature_extractor=L(LdmImplicitCaptionerExtractor)(
        encoder_block_indices=(5, 7),
        unet_block_indices=(2, 5, 8, 11),
        decoder_block_indices=(2, 5),
        steps=(0,),
        learnable_time_embed=True,
        num_timesteps=1,
        clip_model_name="ViT-L-14-336",
    ),
    out_features=["s2", "s3", "s4", "s5"],
    use_checkpoint=True,
    slide_training=True,
)
model.sem_seg_head.pixel_decoder.transformer_in_features = ["s3", "s4", "s5"]
model.clip_head.alpha = 0.35
model.clip_head.beta = 0.65
