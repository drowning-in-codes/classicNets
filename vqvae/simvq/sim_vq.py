from typing import Callable

import torch
from torch import nn
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
        out, = unpack(out, packed_shape, inv_pattern)
        return out

    return packed, inverse


class SimVQ(nn.Module):
    def __init__(self, dim, codebook_size, codebook_transform: Module | None = None, init_fn: Callable = identity,
                 channel_first=False, rotation_trick=True, input_to_quantize_commit_loss_weight=1.,
                 frozen_codebook_dim=None):
        super().__init__()
        self.codebook_size = codebook_size
        self.cha
