from typing import Callable

import torch
from attr.setters import frozen
from torch import nn
from torch.ao.quantization import quantize
from torch.nn import Module
import torch.nn.functional as F

from einx import get_at
from einops import rearrange, pack, unpack


def exists(v):
    return v is not None


def identity(t):
    return t


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        (out,) = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse


def safe_div(num, den, eps=1e-6):
    return num / den.clamp(min=eps)


def l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


# rotation trick related


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, "b d -> b 1 d")
    w = l2norm(u + q, dim=1).detach()

    return (
        e
        - 2 * (e @ rearrange(w, "b d -> b d 1") @ rearrange(w, "b d -> b 1 d"))
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()
            @ rearrange(q, "b d -> b 1 d").detach()
        )
    )


def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, "* d")
    tgt, _ = pack_one(tgt, "* d")

    norm_src = src.norm(dim=-1, keepdim=True)
    norm_tgt = tgt.norm(dim=-1, keepdim=True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src), safe_div(tgt, norm_tgt), src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()

    return inverse(rotated)


class SimVQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_transform: Module | None = None,
        init_fn: Callable = identity,
        channel_first=False,
        rotation_trick=True,
        input_to_quantize_commit_loss_weight=0.25,
        commitment_weight=1.0,
        frozen_codebook_dim=None,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.channel_first = channel_first

        frozen_codebook_dim = default(frozen_codebook_dim, dim)
        codebook = torch.randn(codebook_size, frozen_codebook_dim) * (
            frozen_codebook_dim**-0.5
        )

        codebook = init_fn(codebook)

        if not exists(codebook_transform):
            codebook_transform = nn.Linear(frozen_codebook_dim, dim, bias=False)
        self.code_transform = codebook_transform
        self.register_buffer("frozen_codebook", codebook)

        self.rotation_trick = rotation_trick
        self.input_to_quantize_commit_loss_weight = input_to_quantize_commit_loss_weight
        self.commitment_weight = commitment_weight

    @property
    def codebook(self):
        return self.code_transform(self.frozen_codebook)

    def indices_to_codes(self, indices):
        frozen_codes = self.frozen_codebook(indices)
        quantized = self.code_transform(frozen_codes)

        if self.channel_first:
            quantized = rearrange(quantized, "b ... d -> b d ...")
        return quantized

    def forward(self, x):
        if self.channel_first:
            x = rearrange(x, "b ... d -> b d ...")
        x, inverse_pack = pack_one(x, "b * d")
        implicit_codebook = self.codebook
        with torch.no_grad():
            dist = torch.cdist(x, implicit_codebook)
            indices = dist.argmin(dim=-1)

        quantized = implicit_codebook[indices]
        commit_loss = F.mse_loss(x.detach(), quantized)
        if self.rotation_trick:
            quantized = rotate_to(x, quantized)
        else:
            commit_loss = (
                commit_loss
                + F.mse_loss(x, quantized.detach())
                * self.input_to_quantize_commit_loss_weight
            )
            quantized = (quantized - x).detach() + x
        quantized = inverse_pack(quantized)
        indices = inverse_pack(indices, "b *")
        if self.channel_first:
            quantized = rearrange(quantized, "b ... d-> b d...")
        return quantized, indices, commit_loss * self.commitment_weight
