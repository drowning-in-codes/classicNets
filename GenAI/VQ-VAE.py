#  #!/usr/bin/env python
#  -*- coding:utf-8 -*-
#  Copyleft (C) 2024 proanimer, Inc. All Rights Reserved
#   author:proanimer
#   createTime:2024/9/24 上午11:03
#   lastModifiedTime:2024/9/24 上午11:03
#   file:VQ-VAE.py
#   software: classicNets
#
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, padding=1, ),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, )
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers,
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 4
        stride = 2

        self.conv_stack = nn.Sequential(nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
                                        ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers))

    def forward(self, x):
        return self.conv_stack(x)


class VectorQuantizer(nn.Module):
    """
           Inputs the output of the encoder network z and maps it to a discrete
           one-hot vector that is the index of the closest embedding vector e_j

           z (continuous) -> z_q (discrete)

           z.shape = (batch, channel, height, width)

           quantization pipeline:

               1. get encoder input (B,C,H,W)
               2. flatten input to (B*H*W,C)

           """

    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e  # num_embeddings K
        self.e_dim = e_dim  # embedding_dim H*W*C = D
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_e, 1 / self.n_e)

    def forward(self, z):
        # N C H W-> N H W C
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # calculate distance ||z_e - e_j  || e is the embedding
        # [B*H*W,K]
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2,
                                                                         dim=1) - 2 * torch.matmul(z_flattened,
                                                                                                   self.embedding.weight.t())

        # find the minimal distance index  ranges [0...K-1] K(n_e) is the number of embeddings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  # [B*H*W,1]
        # [B*H*W,K]  # K(n_e) is the number of embeddings  0 or 1
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device=device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)  # [B*H*W,K] * [K,D] = [B*H*W,D]
        # 使用detach,计算时不会影响梯度
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()  # 在反向传播时将z_q的梯度给z

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super().__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
