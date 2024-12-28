import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 7
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        )

    def forward(self, x):
        return self.conv_stack(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_e, 1 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(
            1
        )  # (encoded_feat size,1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device
        )  # (encoded_feat size,embedding_size)
        min_encodings.scatter_(1, min_encoding_indices, 1)  # one-hot  like
        ## 0 0 0 0 0 0 1 0 0 0 0
        ## 0 0 0 0 1 0 0 0 0 0 0
        ## 0 0 0 0 0 0 0 0 1 0 0
        ## 0 0 0 0 0 0 0 1 0 0 0
        ## 0 1 0 0 0 0 0 0 0 0 0
        ## 1 0 0 0 0 0 0 0 0 0 0
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        loss = torch.mean(
            ((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        )
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return loss, z_q, min_encodings, min_encoding_indices


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1
            ),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(
                h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1
            ),
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class VQVAE(nn.Module):
    def __init__(
        self,
        h_dim,
        res_h_dim,
        n_res_layers,
        n_embeddings,
        embedding_dim,
        beta,
        save_img_embedding_map,
    ):
        super(VQVAE, self).__init__()
        # encode the image(3 channels) to h_dim
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        # conv to the embed_dim before quantize
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, 1)
        # quantize the feature
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the quantized feature
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print("original data shape:", x.shape)
            print("encoded data shape:", z_e.shape)
            print("recon data shape:", x_hat.shape)
        return embedding_loss, x_hat, perplexity
