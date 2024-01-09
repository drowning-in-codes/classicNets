import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, input_dim, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc0 = nn.Linear(input_dim, input_dim // ratio, bias=False)
        self.fc1 = nn.Linear(input_dim // ratio, input_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        avg_out = self.fc1(self.fc0(self.avg_pool(x).squeeze()).unsqueeze(2)).unsqueeze(3)
        max_out = self.fc1(self.fc0(self.max_pool(x).squeeze()).unsqueeze(2)).unsqueeze(3)
        out = self.relu(torch.cat([avg_out, max_out], dim=2))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)
