# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
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
