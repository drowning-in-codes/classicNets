#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/2/18 上午10:16
#   lastModifiedTime:2024/2/18 上午10:16
#   file:test.py
#   software: classicNets
#
# compression and denoise model
#
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def spatial_sampling(agent_feature, compress_ratio: float = 0.8):
    """
    spatial sampling except the ego feature
    """
    agent_num, _, H, W = agent_feature.size()
    reduced_pixels = int(H * W * compress_ratio)

    for i in range(
        1, agent_num
    ):  # Start from index 1 to skip the ego feature # agent_feature[i] [C,H,W]
        aggregate_features = torch.sum(agent_feature[i], dim=0)  # [H,W]
        aggregate_features = aggregate_features.reshape(-1)
        _, indices = torch.topk(
            aggregate_features,
            reduced_pixels,
        )

        mask = torch.zeros_like(aggregate_features)
        mask[indices] = 1
        mask = mask.reshape(H, W)
        agent_feature[i] *= mask

    return agent_feature


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout: float = 0.8):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    # TODO: add sr_ratio to reduce the computation cost
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        attn_dropout: float = 0.8,
        proj_dropout: float = 0.8,
        mode="context",
        sr_ratio=None,
        embded_H=None,
        embded_W=None,
        max_seq=5,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.mode = mode
        self.sr_ratio = sr_ratio
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        if sr_ratio is None or sr_ratio == 1:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)
        else:
            assert sr_ratio is not None, "sr_ratio must be set"
            dim = (embded_H // sr_ratio) * (embded_W // sr_ratio)
            self.H = embded_H
            self.W = embded_W
            self.sr = nn.Conv2d(
                max_seq,
                max_seq,
                kernel_size=(sr_ratio, sr_ratio),
                stride=(sr_ratio, sr_ratio),
            )
            self.sr_norm = nn.LayerNorm(dim)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, context=None):
        B, N, C = x.shape
        x = self.norm(x)
        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        if self.sr_ratio is None or self.sr_ratio == 1:
            kv = self.to_kv(x).chunk(2, dim=-1)
            k, v = map(
                lambda t: rearrange(t, " b n (h d)-> b h n d ", h=self.heads), kv
            )
        else:
            x_ = x.reshape(B, N, self.H, self.W)
            x_ = self.sr(x_).reshape(B, N, -1)
            x_ = self.sr_norm(x_)
            kv = self.to_kv(x_).chunk(2, dim=-1)
            k, v = map(lambda t: rearrange(t, "  b n (h d)->b h n d", h=self.heads), kv)

        if context is None:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        else:
            # add context to attention. bias or context mode
            if self.mode == "bias":
                dots = torch.matmul(q, k.transpose(-1, -2))
                dots += self.context
            else:
                dots = torch.matmul(q, k.transpose(-1, -2))  # B,C,
                pos_attn = torch.matmul(q, context.transpose(-1, -2))
                dots += pos_attn

        attn = self.attend(dots)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj_drop(out)
        return self.to_out(out)


class simpleTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth: int,
        dim_head,
        heads=3,
        mlp_dim=None,
        sr_ratio=None,
        embded_H=None,
        embded_W=None,
        max_seq=5,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            sr_ratio=sr_ratio,
                            embded_H=embded_H,
                            embded_W=embded_W,
                            max_seq=max_seq,
                        ),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            x = attn(x, context) + x
            x = ff(x) + x
        return self.norm(x)


device = torch.device("cpu")


class RangeEncoding(nn.Module):
    def __init__(self, max_seq, dim) -> None:
        """
        encode relative position information. e.g. distance,transform_matrix
        """
        super().__init__()
        self.max_seq = max_seq
        self.embedding = nn.Embedding(max_seq, dim)

    def forward(self, prior_info):
        assert prior_info.shape[1] == self.max_seq, "prior encoding shape not correct"
        device = prior_info.device
        prior_info = prior_info.type(torch.long).to(device)

        return self.embedding(prior_info)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_width,
        patch_height,
        feature_width,
        feature_height,
        patch_dim,
        output_dim,
    ):
        """

        :param patch_width:
        :param patch_height:
        :param feature_width:
        :param feature_height:
        :param channels:
        :param output_dim:
        """
        super().__init__()
        assert (feature_width % patch_width == 0) and (
            feature_height % patch_height == 0
        ), "feature dimensions must be divisible by the patch size."
        self.patch_width, self.patch_height = patch_width, patch_height
        self.feature_width, self.feature_height = feature_width, feature_height
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features):
        y = self.to_patch_embedding(features)
        return y


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PEG(nn.Module):
    """
    similar to https://arxiv.org/pdf/2102.10882.pdf
    """

    def __init__(self, dim, k: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(
            dim, dim, kernel_size=k, stride=1, padding=k // 2, groups=dim
        )

    def forward(self, x):
        assert x.ndim == 4, "input tensor must be 4D"  # (1,V,H,W)
        x = self.proj(x) + x
        return x


class simpleConvEmbed(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        output_dim=None,
        dropout: float = 0.5,
    ):
        super().__init__()  # [1,V,H,W]
        if output_dim is None:
            output_dim = patch_size[0] * patch_size[1]
        self.patch_size = patch_size
        feature_h, feature_w = feature_size
        kernel_size = (feature_h // patch_size[0], feature_w // patch_size[1])
        self.depthwise_conv = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            groups=input_dim,
        )  # -> [1,V,H//kernel_size,W//kernel_size]
        self.pointwise_conv = nn.Conv2d(
            input_dim, input_dim, kernel_size=1, stride=1
        )  # -> [1,V,H//kernel_size,W//kernel_size]
        dim = patch_size[0] * patch_size[1]
        self.ln_1 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, output_dim)
        self.ln_2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def get_patch_size(self):
        return self.patch_size

    def forward(self, x):
        assert x.ndim == 3 or x.ndim == 4, "input tensor must be 3D or 4D"
        if x.ndim == 3:
            x = x.unsqueeze(0)
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        out = rearrange(out, "1 V H W -> 1 V (H W)")
        out = self.ln_1(out)
        out = self.fc(out)
        out = self.ln_2(out)
        out = self.dropout(out)
        return out


class ExpandConv(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, groups=output_dim),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class CoVisFormer(nn.Module):
    """
    similar arch with vit and pvt.
    encode distance information to features.  multi-scale
    """

    # patch embedding
    # [N, C, H, W] -> [N, C, H*W] -> [N, H*W, C]
    def __init__(
        self,
        num_vehicles,
        feature_width,
        feature_height,
        channels: int,
        embed_dim=None,
        num_heads: int = 3,
        layer_num=None,
        num_stages: int = 3,
        sr_ratios=None,
    ) -> None:
        super().__init__()
        if sr_ratios is None:
            sr_ratios = [8, 4, 2]
        input_dim = feature_width * feature_height
        if embed_dim is None:
            embed_dim = [
                (feature_height // 4, feature_width // 4),
                (feature_height // 8, feature_width // 8),
                (feature_height // 32, feature_width // 32),
            ]
        if layer_num is None:
            layer_num = [3, 4, 6]
        self.num_stages = num_stages
        num_patchs = num_vehicles  # fixed for all vehicles in the scenario
        self.num_patchs = num_patchs
        for i in range(num_stages):
            # patch_embed = PatchEmbedding(
            #     patch_width,
            #     patch_height,
            #     feature_width,
            #     feature_height,
            #     embed_dim[i - 1] if i > 0 else input_dim,
            #     embed_dim[i],
            # )
            """
            conv embedding first.
            """
            patch_embed = simpleConvEmbed(
                input_dim=num_patchs,
                feature_size=(feature_height, feature_width)
                if i == 0
                else embed_dim[i - 1],
                patch_size=embed_dim[i],
            )
            dim = (
                feature_height * feature_width
                if i == 0
                else embed_dim[i - 1][0] * embed_dim[i - 1][1]
            )
            """
            positional encoding
            """
            pos_embed = RangeEncoding(num_patchs, dim)
            additional_embed = RangeEncoding(num_patchs, dim)
            """
            transformer
            """
            transformerBlock = simpleTransformer(
                dim=dim,
                heads=num_heads,
                dim_head=dim,
                depth=layer_num[i],
                sr_ratio=sr_ratios[i],
                embded_H=embed_dim[i][0],
                embded_W=embed_dim[i][1],
                max_seq=num_patchs,
            )
            """
            PEG model
            """
            peg = PEG(num_patchs, k=3)

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"additional_embed{i + 1}", additional_embed)
            setattr(self, f"block{i + 1}", transformerBlock)
            setattr(self, f"peg{i + 1}", peg)
        """
        expand channel
        """
        self.exconv = ExpandConv(channels)
        self.apply(_init_weights)

    def forward(self, spatial_features, record_len, distances=None):
        if distances is None:
            distances = torch.randn(1, self.num_patchs)
        split_x = self.regroup(
            spatial_features, record_len
        )  # spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x:  # [2,C,H,W]
            """
            将一个batch中的所有车辆的特征当作一个seq,
            """
            V = batch_spatial_feature.shape[0]
            batch_spatial_feature = torch.mean(
                batch_spatial_feature, dim=1
            )  # or sum or max [V,H,W]
            # pad to max_seq.default 5
            batch_spatial_feature = F.pad(
                batch_spatial_feature, [0, 0, 0, 0, self.num_patchs - V, 0]
            )  # [5,H,W]
            for i in range(self.num_stages):
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                pos_embed = getattr(self, f"pos_embed{i + 1}")
                additional_embed = getattr(self, f"additional_embed{i + 1}")
                blks = getattr(self, f"block{i + 1}")
                peg = getattr(self, f"peg{i + 1}")
                batch_spatial_feature = patch_embed(
                    batch_spatial_feature
                )  # [1,V,H,W] -> [1,V,H*W]
                batch_spatial_feature += pos_embed(distances)  # bias mode
                additioanl_info = additional_embed(distances)
                batch_spatial_feature = blks(
                    batch_spatial_feature, additioanl_info
                )  # [1,V,C]
                patch_H, patch_W = patch_embed.get_patch_size()
                batch_spatial_feature = batch_spatial_feature.reshape(
                    1, V, patch_H, patch_W
                )
                batch_spatial_feature = peg(batch_spatial_feature)  # [1,5,H,W]
            batch_spatial_feature = batch_spatial_feature.squeeze(0)[:V].unsqueeze(
                1
            )  # [V,1,H,W]
            batch_spatial_feature = self.exconv(batch_spatial_feature)
            out.append(batch_spatial_feature)
        out = torch.cat(out, dim=0)
        return out

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


if __name__ == "__main__":
    c = torch.randn(4, 256, 200, 256).to(device)
    model = CoVisFormer(
        num_vehicles=2,
        feature_width=200,
        feature_height=256,
        channels=256,
    ).to(device)
    print(
        model(
            c,
            record_len=torch.tensor([2, 2]),
        ).shape
    )
