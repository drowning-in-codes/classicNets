from errno import ELOOP
from functools import partial, cache
from collections import namedtuple

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
import torch.nn.functional as F
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.amp import autocast

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from typing import Callable


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def noop(*args, **kwargs):
    return


def identity(t):
    return t


def l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


def safe_div(num, den, eps=1e-6):
    return num / den.clamp(min=eps)


def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return None
    elif len(modules) == 1:
        return modules[0]
    return nn.Sequential(*modules)


def cdist(x, y):
    x2 = reduce(x**2, "...d->...", "sum")
    y2 = reduce(y**2, "...d->...", "sum")
    xy = einsum("b i d,b j d -> b i j", x, y) * -2
    return (
        (rearrange(x2, "b i -> b i 1") + rearrange(y2, "b j -> b 1 j") + xy)
        .clamp(min=0)
        .sqrt()
    )


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def entropy(prob, eps=1e-5):
    return (-prob * log(prob, eps=eps)).sum(dim=-1)


def ema_inplace(old, new, decay):
    is_mps = str(old.device).startswith("mps:")
    if not is_mps:
        old.lerp_(new, 1 - decay)
    else:
        old.mul_(decay).add_(new * (1 - decay))


def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        (unpacked,) = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def lens_to_mask(lens, max_length):
    seq = torch.arange(max_length, device=lens.device)
    return seq < lens[:, None]


def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform(t)
    return t


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))
