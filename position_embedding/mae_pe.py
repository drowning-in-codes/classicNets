#  #!/usr/bin/env python
#  #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/11 下午9:28
#   lastModifiedTime:2024/6/11 下午9:28
#   file:mae_pe.py
#   software: classicNets
#
import torch
import torch.nn as nn
from absolute_pe import create_1d_absolute_embeddings


def create_2d_absolute_sincos_embeddings(height, width, dim):
    assert dim % 4 == 0
    position_embedding = torch.zeros(height * width, dim)
    coords = torch.stack(torch.meshgrid(torch.arange(height, dtype=torch.float),
                                        torch.arange(width, dtype=torch.float)))  # [2,height,width]
    height_embedding = create_1d_absolute_embeddings(torch.flatten(coords[0]), dim // 2)  # [height*width,dim//2]
    width_embedding = create_1d_absolute_embeddings(torch.flatten(coords[1]), dim // 2)  # [height*width,dim//2]
    position_embedding[:, :dim // 2] = height_embedding
    position_embedding[:, dim // 2:] = width_embedding
    return position_embedding
