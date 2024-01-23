from typing import Optional, Union
from collections import OrderedDict
import torch
import torch.nn as nn

class SKConv(nn.Module):
    """
    https://arxiv.org/pdf/1903.06586.pdf
    """

    def __init__(self, feature_dim, WH, M, G, r, stride=1, L=32):

        """ Constructor
         Args:
             features: input channel dimensionality.
             WH: input spatial dimensionality, used for GAP kernel size.
             M: the number of branchs.
             G: num of convolution groups.
             r: the radio for compute d, the length of z.
             stride: stride, default 1.
             L: the minimum dim of the vector z in paper, default 32.
        """
        super().__init__()
        d = max(int(feature_dim / r), L)
        self.M = M
        self.feature_dim = feature_dim
        self.convs = nn.ModuleList()
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=False)
            ))
        self.gap = nn.AdaptiveAvgPool2d(int(WH/stride))
        self.fc = nn.Linear(feature_dim, d)
        self.fcs = nn.ModuleList()
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, feature_dim)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            feat = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = feat
            else:
                feas = torch.cat((feas, feat), dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat((attention_vectors, vector), dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas*attention_vectors).sum(dim=1)
        return fea_v

class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
           Args:
               in_features: input channel dimensionality.
               out_features: output channel dimensionality.
               WH: input spatial dimensionality, used for GAP kernel size.
               M: the number of branchs.
               G: num of convolution groups.
               r: the radio for compute d, the length of z.
               mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
               stride: stride.
               L: the minimum dim of the vector z in paper.

   ————————————————
        """
        super().__init__()
        super().__int__()
        if mid_features is None:
            mid_features = int(out_features//2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride, L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1),
            nn.BatchNorm2d(out_features)
        )

        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1),
                nn.BatchNorm2d(out_features)
            )

    def forward(self,x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

class SKNet(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.BatchNorm2d(64)
        )
        self.stage_1 = nn.Sequential(
            SKUnit(64, 256, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 8, 1),
            nn.ReLU(),
            SKUnit(256, 256, 32, 2, 2, 1),
            nn.ReLU(),
        )
        self.stage_2 = nn.Sequential(
            SKUnit(256, 512, 16, 2, 8, 2),
            nn.ReLU(),
            SKUnit(512, 512, 16, 2, 8, 1),
            nn.ReLU(),
            SKUnit(512, 512, 16, 2, 2, 1),
            nn.ReLU(),
        )
        self.stage_3 = nn.Sequential(
            SKUnit(512, 1024, 8, 2, 8, 2),
            nn.ReLU(),
            SKUnit(1024, 1024, 8, 2, 8, 1),
            nn.ReLU(),
            SKUnit(1024, 1024, 8, 2, 2, 1),
            nn.ReLU(),
        )

        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(1024, class_num)

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea

