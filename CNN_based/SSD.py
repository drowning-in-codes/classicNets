#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/12 下午10:42
#   lastModifiedTime:2024/6/12 下午10:42
#   file:SSD.py
#   software: classicNets
#
import torch
import torch.nn as nn
from torchvision.models import vgg19


class SSD(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        vgg.eval()
        self.conv = vgg.features
        self.conv1 = nn.Conv2d(512, 1024, 3)
        self.conv2 = nn.Conv2d(1024, 1024, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(256, 512, 3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(128, 256, 3)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 256, 3)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 256, 3)

        )

    def forward(self, x):
        feat_1 = self.conv(x)
        feat = self.conv1(feat_1)
        feat_2 = self.conv2(feat)
        feat_3 = self.conv3(feat_2)
        feat_4 = self.conv4(feat_3)
        feat_5 = self.conv5(feat_4)
        feat_6 = self.conv6(feat_5)
        return feat_1, feat_2, feat_3, feat_4, feat_5, feat_6
