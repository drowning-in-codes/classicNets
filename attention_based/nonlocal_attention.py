import torch
import torch.nn as nn


class NonLocalNet(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(NonLocalNet, self).__init__()
        intermediate_dim = input_dim // 2
        self.to_q = nn.Conv2d(input_dim, intermediate_dim, 1)
        self.to_k = nn.Conv2d(input_dim, intermediate_dim, 1)
        self.to_v = nn.Conv2d(input_dim, intermediate_dim, 1)

        self.conv = nn.Conv2d(intermediate_dim, output_dim, 1)

    def forward(self, x):
        q = self.to_q(x).squeeze()
        k = self.to_k(x).squeeze()
        v = self.to_v(x).squeeze()

        u = torch.bmm(q, k.transpose(1, 2))
        u = torch.softmax(u, dim=1)
        out = torch.bmm(u, v)
        out = out.unsqueeze(2)
        out = self.conv(out)
        return out + x
