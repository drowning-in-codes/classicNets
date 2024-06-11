#   #!/usr/bin/env python
#   #-*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/6/11 下午2:35
#   lastModifiedTime:2024/6/11 下午2:35
#   file:relative_pe.py
#   software: classicNets
# from https://arxiv.org/pdf/1803.02155

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None:] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        # [0,2k] 2k+1 total positions
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(DEVICE)
        embeddings = self.embeddings_table[final_mat].to(DEVICE)

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2
        self.relative_position_k = RelativePosition(
            self.head_dim, self.max_relative_position
        )
        self.relative_position_v = RelativePosition(
            self.head_dim, self.max_relative_position
        )

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None,
    ):
        batch_size = query.shape[0]
        len_q = query.shape[1]
        len_k = key.shape[1]
        len_v = value.shape[1]
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))  # q*k
        r_q2 = (
            query.permute(1, 0, 2)
            .contiguous()
            .view(len_q, batch_size * self.n_heads, self.head_dim)
        )
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)  # q*pos
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale  # q*k + q*pos  k上加位置编码

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        weight1 = torch.matmul(attn, r_v1)  # a*v  a=softmax(q*k + q*pos)

        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)  # a*pos_v

        weight2 = weight2.transpose(0, 1).reshape(
            batch_size, self.n_heads, len_q, self.head_dim
        )
        x = weight1 + weight2
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x


if __name__ == '__main__':
    model = MultiHeadAttentionLayer(512, 8, 0.1, DEVICE)
    model = model.to(DEVICE)
    query = torch.rand(64, 10, 512).to(DEVICE)
    key = torch.rand(64, 10, 512).to(DEVICE)
    value = torch.rand(64, 10, 512).to(DEVICE)
    # mask = torch.rand(64, 10, 20).to(DEVICE)
    output = model(query, key, value)
    print(output.shape)
