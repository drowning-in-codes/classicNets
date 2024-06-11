#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/11 下午7:19
#   lastModifiedTime:2024/6/11 下午7:19
#   file:absolute_pe.py
#   software: classicNets
#
import torch
import torch.nn as nn
import math


def create_1d_absolute_embeddings(n_pos_vec, dim):
    assert dim % 2 == 0
    pos_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)
    omega = torch.arange(dim // 2, dtype=torch.float)
    omega /= dim / 2.
    omega = torch.pow(10000, omega)
    out = n_pos_vec[:, None] @ omega[None, :]
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    pos_embedding[:, 0::2] = emb_sin
    pos_embedding[:, 1::2] = emb_cos
    return pos_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层

        # 计算位置编码并将其存储在pe张量中
        pe = torch.zeros(max_len, d_model)  # 创建一个max_len x d_model的全零张量
        position = torch.arange(0, max_len).unsqueeze(1)  # 生成0到max_len-1的整数序列，并添加一个维度
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在第一个维度添加一个维度，以便进行批处理
        self.register_buffer('pe', pe)  # 将位置编码张量注册为缓冲区，以便在不同设备之间传输模型时保持其状态

    # 定义前向传播函数
    def forward(self, x):
        # 将输入x与对应的位置编码相加
        x = x + self.pe[:, :x.size(1)]

        # 应用dropout层并返回结果
        return self.dropout(x)
