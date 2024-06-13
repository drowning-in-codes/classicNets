#!/usr/bin/env python
# -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/12 下午8:28
#   lastModifiedTime:2024/6/12 下午8:28
#   file:UNet.py
#   software: classicNets
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channel, output_channel, is_downsamping=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.ReLU())
        self.downsample = nn.MaxPool2d(2, 2, 0) if is_downsamping else nn.Identity()

    def forward(self, x):
        before_downsample = self.conv2(self.conv1(x))
        output = self.downsample(before_downsample)
        return before_downsample, output


class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, is_upsamping=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.upsample = nn.ConvTranspose2d(output_channel, output_channel, 2, 2, 0) if is_upsamping else nn.Identity()

    def forward(self, x):
        before_upsample = self.conv2(self.conv1(x))
        output = self.upsample(before_upsample)
        return before_upsample, output


class Header(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, 1),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, 3, 1),
            nn.ReLU()
        )
        # self.register_buffer() # 不变参数,但是会存在state_dict中
        # self.register_parameter() # 类似 nn.Parameter
        # self.register_module() # module,跟继承了nn.Module的类类似

        self.head = nn.Conv2d(output_channel, 2, 1, 1)

    def forward(self, x):
        return self.head(self.conv(x))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_encoder = Encoder(1, 64)
        self.encoder1 = Encoder(64, 128)
        self.encoder2 = Encoder(128, 256)
        self.encoder3 = Encoder(256, 512)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)

        self.header = Header(64, 1)

    def forward(self, x):
        result_1, downsample_result_1 = self.input_encoder(x)
        result_2, downsample_result_2 = self.encoder1(downsample_result_1)
        result_3, downsample_result_3 = self.encoder2(downsample_result_2)
        result_4, downsample_result_4 = self.encoder3(downsample_result_3)

        _, upsample_result = self.decoder1(downsample_result_4)
        result_4 = F.interpolate(result_4, size=upsample_result.size()[2:], mode='bilinear', align_corners=True)
        _, upsample_result = self.decoder2(torch.cat([upsample_result, result_4], dim=1))
        result_3 = F.interpolate(result_3, size=upsample_result.size()[2:], mode='bilinear', align_corners=True)
        _, upsample_result = self.decoder3(torch.cat([upsample_result, result_3]))
        result_2 = F.interpolate(result_2, size=upsample_result.size()[2:], mode='bilinear', align_corners=True)
        _, upsample_result = self.decoder4(torch.cat([upsample_result, result_2]))
        result_1 = F.interpolate(result_1, size=upsample_result.size()[2:], mode='bilinear', align_corners=True)
        result = self.header(torch.cat([upsample_result, result_1]))
        return result
