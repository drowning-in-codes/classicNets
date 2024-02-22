#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/2/22 下午9:25
#   lastModifiedTime:2024/2/22 下午9:25
#   file:LeNet.py
#   software: classicNets
#

import torch
import torch.nn as nn
import torch.nn.functional as F


# LeNet for MNIST
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        out = F.relu(self.conv1(x)) # 3*32*32 -> 6*28*28
        out = F.max_pool2d(out,2) # 6*28*28 -> 6*14*14
        out = F.relu(self.conv2(out)) # 6*14*14 -> 16*10*10
        out = F.max_pool2d(out, 2) # 16*10*10 -> 16*5*5
        out = out.view(out.size(0), -1) # 16*5*5 -> 400

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 3*224*224 -> 64*55*55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*55*55 -> 64*27*27
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # 64*27*27 -> 192*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 192*27*27 -> 192*13*13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 192*13*13 -> 384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 384*13*13 -> 256*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256*13*13 -> 256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256*13*13 -> 256*6*6
        )
        self.fc = nn.Linear(256,10)

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out