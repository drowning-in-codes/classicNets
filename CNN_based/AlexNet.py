#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/2/22 下午9:25
#   lastModifiedTime:2024/2/22 下午9:25
#   file:AlexNet.py
#   software: classicNets
#

import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 3*224*224 -> 64*55*55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*55*55 -> 64*27*27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 64*27*27 -> 192*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 192*27*27 -> 192*13*13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 192*13*13 -> 384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 384*13*13 -> 256*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*13*13 -> 256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256*13*13 -> 256*6*6
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out