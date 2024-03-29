#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/2/18 上午10:38
#   lastModifiedTime:2024/2/18 上午10:38
#   file:cvt.py
#   software: classicNets
#
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum


class SepConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
    ):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):
    def __init__(
        self,
        dim,
        img_size,
        heads=8,
        dim_head=64,
        kernel_size=3,
        q_stride=1,
        k_stride=1,
        v_stride=1,
        dropout=0.0,
        last_stage=False,
    ):
        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), "b n (h d) -> b h n d", h=h)
        x = rearrange(x, "b (l w) n -> b n l w", l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, "b (h d) l w -> b h (l w) d", h=h)

        v = self.to_v(x)
        v = rearrange(v, "b (h d) l w -> b h (l w) d", h=h)

        k = self.to_k(x)
        k = rearrange(k, "b (h d) l w -> b h (l w) d", h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        img_size,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        last_stage=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            ConvAttention(
                                dim,
                                img_size,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                last_stage=last_stage,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class cvt(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        num_classes,
        dim=64,
        kernels=[7, 3, 3],
        strides=[4, 2, 2],
        heads=[1, 3, 6],
        depth=[1, 2, 10],
        pool="cls",
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
    ):
        super(cvt, self).__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
        self.dim = dim
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 2),
            Rearrange("b c h w -> b (h w) c", h=image_size // 4, w=image_size // 4),
            nn.LayerNorm(dim),
        )
        self.stage_1_transformer = nn.Sequential(
            Transformer(
                dim,
                img_size=image_size // 4,
                depth=depth[0],
                heads=heads[0],
                dim_head=dim // heads[0],
                mlp_dim=dim * scale_dim,
                dropout=dropout,
                last_stage=True,
            ),
            Rearrange("b (h w) c -> b c h w", h=image_size // 4, w=image_size // 4),
        )
        #     stage 2
        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange("b c h w -> b (h w) c", h=image_size // 8, w=image_size // 8),
            nn.LayerNorm(dim),
        )
        self.stage_2_transformer = nn.Sequential(
            Transformer(
                dim,
                img_size=image_size // 8,
                depth=depth[1],
                heads=heads[1],
                dim_head=dim // heads[1],
                mlp_dim=dim * scale_dim,
                dropout=dropout,
                last_stage=True,
            ),
            Rearrange("b (h w) c -> b c h w", h=image_size // 8, w=image_size // 8),
        )
        #     stage 3
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
            Rearrange("b c h w -> b (h w) c", h=image_size // 16, w=image_size // 16),
            nn.LayerNorm(dim),
        )
        self.stage_3_transformer = nn.Sequential(
            Transformer(
                dim=dim,
                img_size=image_size // 16,
                depth=depth[2],
                heads=heads[2],
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout,
                last_stage=True,
            ),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.drop_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        xs = self.stage1_conv_embed(img)
        xs = self.stage1_transformer(xs)

        xs = self.stage2_conv_embed(xs)
        xs = self.stage2_transformer(xs)

        xs = self.stage3_conv_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage_3_transformer(xs)
        xs = xs.mean(dim=1) if self.pool == "mean" else xs[:, 0]

        xs = self.mlp_head(xs)
        return xs
