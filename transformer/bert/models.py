#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/13 下午9:12
#   lastModifiedTime:2024/6/13 下午9:12
#   file:models.py
#   software: classicNets
#
import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last


class Config(NamedTuple):
    """configuration for BERT model"""
    vocab_size: int = None  # Size of Vocabulary
    dim: int = 768  # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12  # Numher of Hidden Layers
    n_heads: int = 12  # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768 * 4  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1  # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1  # Probability of Dropout of Attention Layers
    max_len: int = 512  # Maximum Length for Positional Embeddings
    n_segments: int = 2  # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        obj = json.load(open(file, "r"))
        return cls(**obj)


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)

        self.drop = nn.Dropout(cfg.p_drop_attn)

        self.scores = None
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x),
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward NN for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ transformer block """

    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_attn)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(x + self.drop(self.pwff(h)))
        return h


class Embeddings(nn.Module):
    """ the embedding module from word,position and token_type embedding"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
       """
        seg_len = x.size(1)
        pos = torch.arange(seg_len, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class Transformer(nn.Module):
    """ transformer with self-attentive blocks """

    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h
