import torch
import torch.nn as nn
import math


# implement self attention using scaled dot product
class SelfAttention(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.scale = scale if scale is None else math.sqrt(64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v):
        # q: [batch_size, num_heads, seq_len, input_dim]
        # k: [batch_size, num_heads, seq_len, input_dim]
        # v: [batch_size, num_heads, seq_len, input_dim]
        # out: [batch_size, num_heads, seq_len, input_dim]
        u = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attention = self.softmax(u)
        out = torch.bmm(attention, v)
        return out

    @torch.no_grad()
    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# implement multi-head attention
class MultiHeadAttention(nn.Module):
    # implement multi-head attention
    def __init__(self, num_heads: int, scale: float):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.fc_q = nn.Linear(64, 64 * num_heads)
        self.fc_k = nn.Linear(64, 64 * num_heads)
        self.fc_v = nn.Linear(64, 64 * num_heads)
        self.attention = SelfAttention(scale)
        self.fc = nn.Linear(64, 64)

    def forward(self, q, k, v):
        # q: [batch_size, seq_len, input_dim]
        # k: [batch_size, seq_len, input_dim]
        # v: [batch_size, seq_len, input_dim]
        # out: [batch_size, seq_len, input_dim]
        batch_size = q.size(0)
        q = self.fc_q(q)
        k = self.fc_k(k)

        v = self.fc_v(v)
        q = q.view(batch_size, -1, self.num_heads, 64 // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, 64 // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, 64 // self.num_heads).transpose(1, 2)

        out = self.attention(q, k, v)
        # split head and concat
        out = out.transpose(1, 2).contiguous().reshape(batch_size, -1, 64)
        out = self.fc(out)

        return out
