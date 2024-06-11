#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/11 下午7:31
#   lastModifiedTime:2024/6/11 下午7:31
#   file:vit_pe.py
#   software: classicNets
#
import torch
import torch.nn as nn


def create_1d_absolute_trainable_embeddings(max_len, dim):
    assert dim % 2 == 0
    pos_embedding = nn.Embedding(max_len, dim)
    nn.init.xavier_uniform_(pos_embedding.weight)
    return pos_embedding
