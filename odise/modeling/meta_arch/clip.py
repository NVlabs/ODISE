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
from collections import OrderedDict, namedtuple
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from detectron2.utils import comm
from einops import rearrange

from .helper import ensemble_logits_with_labels

logger = logging.getLogger(__name__)

EmbeddedText = namedtuple("EmbedTextReturn", ["text_embed", "text_encodings", "text_mask"])
EmbeddedImage = namedtuple("EmbedImageReturn", ["image_embed", "image_encodings"])


def build_clip_text_embed(clip_model_name, labels, device="cuda", verbose=True):
    if isinstance(clip_model_name, str):
        clip, _, _ = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained="openai",
            device=device if torch.cuda.is_available() else "cpu",
        )
        if verbose:
            logger.info(f"Loading CLIP model {clip_model_name}")
    else:
        clip = clip_model_name
        if verbose:
            logger.info("Using provided CLIP model")
    clip_device = next(clip.parameters()).device
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels[0], str):
        labels = [[t] for t in labels]

    labels = tuple(tuple(t) for t in labels)

    # check if is ensemble
    assert isinstance(
        labels[0], (list, tuple)
    ), f"labels should be a list of list of str, but got {type(labels[0])}"

    # unravel list of list of str
    flatten_text = [t for sublist in labels for t in sublist]

    text_embed_list = []

    local_batch_size = 256

    for i in range(0, len(flatten_text), local_batch_size):
        cur_text = flatten_text[i : i + local_batch_size]
        text_embed = clip.encode_text(open_clip.tokenize(cur_text).to(clip_device))
        text_embed_list.extend(list(text_embed))

    out_text_embed = torch.stack(text_embed_list)
    if verbose:
        logger.info(
            f"Built text_embed of shape {out_text_embed.shape} for {len(labels)} labels: {labels}"  # noqa
        )

    return out_text_embed


# Modified from https://github.com/lucidrains/DALLE2-pytorch/blob/350a3d60456693a8ecdccc820e97dbb6b0c81866/dalle2_pytorch/dalle2_pytorch.py#L238 # noqa
class ClipAdapter(nn.Module):
    def __init__(self, name="ViT-B-32", normalize=True):

        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(name, pretrained="openai")
        comm.synchronize()

        # checked, the same as openai original CLIP
        openai_clip, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained="openai"
        )
        super().__init__()
        self.clip = openai_clip

        # self.clip_normalize = preprocess.transforms[-1]
        # the first two are Resize and Crop, the last one is normalization
        self.clip_preprocess = T.Compose([*preprocess.transforms[:2], preprocess.transforms[-1]])
        self._freeze()
        self.name = name
        self.normalize = normalize

    def extra_repr(self) -> str:
        return f"name={self.name}, normalize={self.normalize}"

    def _freeze(self):
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def ignored_state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, "ignored_state_dict"):
                module.ignored_state_dict(destination, prefix + name + ".")
        return super().state_dict(destination=destination, prefix=prefix)

    @property
    def device(self):
        return next(self.parameters()).device

    # don't save clip model
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return OrderedDict()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze()
        return self

    @property
    def dim_latent(self):
        return self.clip.text_projection.shape[-1]

    @property
    def image_size(self):
        if isinstance(self.clip.visual.image_size, tuple):
            return self.clip.visual.image_size
        else:
            return (self.clip.visual.image_size, self.clip.visual.image_size)

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    def _encode_text(self, text):
        x = self.clip.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)
        text_encodings = x

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        return text_embed, text_encodings

    @torch.no_grad()
    def embed_text(self, captions):
        text = open_clip.tokenize(captions).to(next(self.parameters()).device)
        text = text[..., : self.max_text_len]
        text_mask = (text != 0).long()

        text_embed, text_encodings = self._encode_text(text)
        if self.normalize:
            return EmbeddedText(
                F.normalize(text_embed.float(), dim=-1), text_encodings.float(), text_mask
            )
        else:
            return EmbeddedText(text_embed.float(), text_encodings.float(), text_mask)

    def _encode_image(self, image):
        if hasattr(self.clip.visual, "positional_embedding"):
            x = self.clip.visual.conv1(image)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.clip.visual.class_embedding.to(x.dtype)
                    + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + self.clip.visual.positional_embedding.to(x.dtype)
            x = self.clip.visual.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            # [batch_size, num_patches+1, transformer.width]
            x = self.clip.visual.ln_post(x)
            batch_size, num_tokens, _ = x.shape

            if self.clip.visual.proj is not None:
                x = rearrange(x, "b n c -> (b n) c", b=batch_size, n=num_tokens)
                x = x @ self.clip.visual.proj
                x = rearrange(x, "(b n) c -> b n c", b=batch_size, n=num_tokens)

            image_embed = x[:, 0, :]
            image_encodings = x[:, 1:, :]

            width = height = int(image_encodings.shape[1] ** 0.5)

            image_encodings = rearrange(image_encodings, "b (h w) c -> b c h w", h=height, w=width)

            image_encodings = F.interpolate(
                image_encodings,
                size=(image.shape[2] // 16, image.shape[3] // 16),
                mode="bilinear",
                align_corners=False,
            )

            return image_embed, image_encodings
        else:
            image_embed = self.clip.encode_image(image)
            return image_embed, None

    @torch.no_grad()
    def embed_image(self, image):
        image_embed, image_encodings = self._encode_image(self.clip_preprocess(image))
        if self.normalize:
            return EmbeddedImage(F.normalize(image_embed.float(), dim=-1), image_encodings)
        else:
            return EmbeddedImage(image_embed.float(), image_encodings)

    @torch.no_grad()
    def build_text_embed(self, labels):
        return build_clip_text_embed(self.clip, labels)


# Thanks Zheng Ding for sharing the nice implementation, we modified based on that.
class MaskCLIP(ClipAdapter):
    """
    Ref: https://arxiv.org/abs/2208.08984
    """

    def __init__(self, name="ViT-L-14-336"):
        super().__init__(name=name, normalize=False)

    @property
    def logit_scale(self):
        logit_scale = torch.clamp(self.clip.logit_scale.exp(), max=100)
        return logit_scale

    def _mask_clip_forward(self, x: torch.Tensor, attn_mask: torch.Tensor, num_mask_tokens: int):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.clip.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        cls_embed = x[0:1]
        cls_embed = cls_embed.expand(num_mask_tokens, -1, -1)
        x = torch.cat([cls_embed, x], dim=0)
        x = self.clip.visual.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # [N, L, D]
        x = self.clip.visual.ln_post(x[:, :num_mask_tokens, :])

        if self.clip.visual.proj is not None:
            x = torch.einsum("nld,dc->nlc", x, self.clip.visual.proj)

        return x

    def encode_image_with_mask(self, image, mask):
        assert hasattr(self.clip.visual, "positional_embedding")
        image = self.clip_preprocess(image)
        batch_size = image.shape[0]
        assert batch_size == mask.shape[0]
        num_queries = mask.shape[1]

        # [B, Q, H, W], Q is the number of quries, H and W are the height and width of the image
        mask = mask.sigmoid()
        # [B, Q, H//P, W//P]
        patch_mask = F.max_pool2d(
            mask,
            kernel_size=self.clip.visual.conv1.kernel_size,
            stride=self.clip.visual.conv1.stride,
        )
        # 0 means not masked out, 1 mean masked out
        # so if 1 pixel > 0.5, it is not masked out
        # aka if all pixels (max pixel) < 0.5, it is masked out
        mask_token_attn_mask = patch_mask < 0.5
        # [B, Q, H//P x W//P]
        mask_token_attn_mask = mask_token_attn_mask.reshape(batch_size, num_queries, -1)

        num_mask_token = num_queries
        num_image_cls_token = self.clip.visual.positional_embedding.shape[0]
        num_image_token = num_image_cls_token - 1
        num_all_token = num_mask_token + num_image_cls_token

        # we start with no mask out
        attn_mask = torch.zeros(
            (num_all_token, num_all_token), dtype=torch.bool, device=image.device
        )

        # mask+cls+image token to mask token attention is masked out
        attn_mask[:, :num_mask_token] = True

        attn_mask = attn_mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        attn_mask[:, :num_mask_token, -num_image_token:] = mask_token_attn_mask
        num_heads = self.clip.visual.conv1.out_channels // 64  # head width 64
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * num_heads, num_all_token, num_all_token)

        return self._mask_clip_forward(image, attn_mask, num_mask_token)

    def get_mask_embed(self, image, mask):

        image = F.interpolate(
            image,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)

        # [B, Q, C]
        mask_embed = self.encode_image_with_mask(image, mask)

        return mask_embed

    def pred_logits(self, mask_embed, text_embed, labels):
        logit_per_mask = (
            torch.einsum(
                "bqc,nc->bqn", F.normalize(mask_embed, dim=-1), F.normalize(text_embed, dim=-1)
            )
            * self.logit_scale
        )

        logit_per_mask = ensemble_logits_with_labels(logit_per_mask, labels)

        return logit_per_mask

    def forward(self, image, mask, text_embed, labels):

        mask_embed = self.get_mask_embed(image, mask)
        output = {"mask_embed": mask_embed}

        if text_embed is not None and labels is not None:

            output["mask_pred_open_logits"] = self.pred_logits(mask_embed, text_embed, labels)

        return output
