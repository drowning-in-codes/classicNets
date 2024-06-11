#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/3/6 下午10:25
#   lastModifiedTime:2024/3/6 下午10:25
#   file:stand-alone-self-attention.py
#   software: classicNets
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        groups=1,
        bias=False,
    ):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert (
            self.out_channels % self.groups == 0
        ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"
        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1))
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size))

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, height, width, -1
        )
        v_out = v_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, height, width, -1
        )

        q_out = q_out.view(
            batch, self.groups, self.out_channels // self.groups, height, width, 1
        )
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum("bnchwv,bnchwv->bnchw", out, v_out).reshape(
            batch, -1, height, width
        )
        return out


class AttentionStem(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        m=4,
        bias=False,
    ):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert (
            self.out_channels % self.groups == 0
        ), "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(
            torch.randn(out_channels // groups, kernel_size), requires_grad=True
        )
        self.emb_b = nn.Parameter(
            torch.randn(out_channels // groups, kernel_size), requires_grad=True
        )
        self.emb_mix = nn.Parameter(
            torch.randn(m, out_channels // groups), requires_grad=True
        )

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
                for _ in range(m)
            ]
        )

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack(
            [self.value_conv[_](padded_x) for _ in range(self.m)], dim=0
        )

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(
            4, self.kernel_size, self.stride
        )

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum("mc,ca->ma", self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum("mc,cb->mb", self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(
            self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size
        )

        v_out = emb * v_out

        k_out = k_out.contiguous().view(
            batch, self.groups, self.out_channels // self.groups, height, width, -1
        )
        v_out = v_out.contiguous().view(
            self.m,
            batch,
            self.groups,
            self.out_channels // self.groups,
            height,
            width,
            -1,
        )
        v_out = torch.sum(v_out, dim=0).view(
            batch, self.groups, self.out_channels // self.groups, height, width, -1
        )

        q_out = q_out.view(
            batch, self.groups, self.out_channels // self.groups, height, width, 1
        )

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum("bnchwk,bnchwk->bnchw", out, v_out).view(
            batch, -1, height, width
        )

        return out
