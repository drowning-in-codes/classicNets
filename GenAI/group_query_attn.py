#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/10/9 10:48
#   lastModifiedTime:2024/10/9 10:48
#   file:group_query_attn.py
#   software: classicNets
#
from typing import Optional

from einops import rearrange
from torch import Tensor


def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    force_grouped: bool = False,
):
    """
    scaled dot product attention with support for grouped queries
    :param query:
    :param key:
    :param value:
    :param dropout:
    :param scale:
    :param mask:
    :param is_causal:
    :param need_weights:
    :param average_attn_weights:
    :param force_grouped:
    :return:
    """
    if mask is not None and is_causal is not None:
        raise ValueError("mask and is_causal cannot be used together")
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")
