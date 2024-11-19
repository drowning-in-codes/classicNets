import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv(x).squeeze()
        return x


import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv4.weight)


# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)
from torch.autograd import Function


class _GradReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, constant):
        assert isinstance(constant, int) and constant > 0
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


class GradReverseLayer(nn.Module):
    def __init__(self, weight):
        super(GradReverseLayer, self).__init__()
        self.weight = weight

    def forward(self, x):
        return _GradReverseLayer.apply(x, self.weight)


def backward_hook(module, grad_input, grad_output):
    output_grad_input = -grad_input[0]
    return (output_grad_input,)


class negGradient(nn.Module):
    def __init__(self):
        super(negGradient, self).__init__()
        self.register_full_backward_hook(backward_hook)

    def forward(self, x):
        return x


import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
from typing import Tuple


class ReZero(nn.Module):
    def __init__(self, in_channels: int, res_channels: int):
        super(ReZero, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, res_channels: int, nb_layers: int):
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(*[ReZero(in_channels, res_channels)
                                     for _ in range(nb_layers)
                                     ])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int, hidden_channels: int,
                 res_channels: int, nb_res_layers: int,
                 downscale_factor: int,
                 ):
        super(Encoder, self).__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int, hidden_channels: int, out_channels: int,
                 res_channels: int, nb_res_layers: int,
                 upscale_factor: int,
                 ):
        super(Decoder, self).__init__()
        assert log2(upscale_factor) % 1 == 0, "Downscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


"""
    Almost directly taken from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py
    No reason to reinvent this rather complex mechanism.

    Essentially handles the "discrete" part of the network, and training through EMA rather than 
    third term in loss function.
"""


class CodeLayer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, nb_entries: int):
        super(CodeLayer, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)

        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.conv_in(x.float()).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)

        # Calculate distances between input and embeddings
        # TODO: add feature selection stage. use ego query

        # TODO: add cos similarity
        # cos_d = F.cosine_similarity(flatten.unsqueeze(1), self.embed.t().unsqueeze(0), dim=2)
        # TODO: add mi loss

        # TODO: multiple codes per feature? multi-head codebook

        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )  # shape: (flatten.shape,embedding_shape) (H*W,1200)
        # dist = dist + cos_d
        _, embed_ind = (-dist).max(1)  # shape (H*W,1200)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # shape (H*W,1200)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)  # shape (1200)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        #  TODO: add mutual information loss using KL/JS divergence

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Upscaler(nn.Module):
    def __init__(self, embed_dim: int, scaling_rates):
        super(Upscaler, self).__init__()

        self.stages = nn.ModuleList()
        for sr in scaling_rates:
            upscale_steps = int(log2(sr))
            layers = []
            for _ in range(upscale_steps):
                layers.append(nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(embed_dim))
                layers.append(nn.ReLU(inplace=True))
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: torch.FloatTensor, stage: int) -> torch.FloatTensor:
        return self.stages[stage](x)


"""
    Main VQ-VAE-2 Module, capable of support arbitrary number of levels
    TODO: A lot of this class could do with a refactor. It works, but at what cost?
    TODO: Add disrete code decoding function
"""


class MVQVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_channels: int = 128,
                 res_channels: int = 32,
                 nb_res_layers: int = 2,
                 nb_levels: int = 3,
                 embed_dim: int = 64,
                 nb_entries: int = 512,
                 scaling_rates=[8, 4, 2]
                 ):
        super(MVQVAE, self).__init__()
        self.nb_levels = nb_levels
        assert len(scaling_rates) == nb_levels, "Number of scaling rates not equal to number of levels!"

        self.encoders = nn.ModuleList(
            [Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))
        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(CodeLayer(hidden_channels + embed_dim, embed_dim, nb_entries))
        self.codebooks.append(CodeLayer(hidden_channels, embed_dim, nb_entries))

        self.decoders = nn.ModuleList([Decoder(embed_dim * nb_levels, hidden_channels, in_channels, res_channels,
                                               nb_res_layers, scaling_rates[0])])
        for i, sr in enumerate(scaling_rates[1:]):
            self.decoders.append(
                Decoder(embed_dim * (nb_levels - 1 - i), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

        self.upscalers = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.upscalers.append(Upscaler(embed_dim, scaling_rates[1:len(scaling_rates) - i][::-1]))

    def forward(self, x, verbose=False):
        encoder_outputs = []
        code_outputs = []
        decoder_outputs = []
        upscale_counts = []
        id_outputs = []
        diffs = []

        for enc_index, enc in enumerate(self.encoders):
            enc_output = None
            if len(encoder_outputs):
                enc_output = enc(encoder_outputs[-1])
                encoder_outputs.append(enc_output)
            else:
                enc_output = enc(x)
                encoder_outputs.append(enc(x))
            if verbose:
                print("encode layer", enc_index, "encode shape", enc_output.shape)

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]

            if len(decoder_outputs):  # if we have previous levels to condition on
                code_q, code_d, emb_id = codebook(torch.cat([encoder_outputs[l], decoder_outputs[-1]], axis=1))
            else:
                code_q, code_d, emb_id = codebook(encoder_outputs[l])
            diffs.append(code_d)
            id_outputs.append(emb_id)

            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        # return decoder_outputs[-1], diffs, encoder_outputs, decoder_outputs, id_outputs
        for decode_output in decoder_outputs:
            print("decode output shape", decode_output.shape)
        return diffs, decoder_outputs[-1], None

    def decode_codes(self, *cs):
        decoder_outputs = []
        code_outputs = []
        upscale_counts = []

        for l in range(self.nb_levels - 1, -1, -1):
            codebook, decoder = self.codebooks[l], self.decoders[l]
            code_q = codebook.embed_code(cs[l]).permute(0, 3, 1, 2)
            code_outputs = [self.upscalers[i](c, upscale_counts[i]) for i, c in enumerate(code_outputs)]
            upscale_counts = [u + 1 for u in upscale_counts]
            decoder_outputs.append(decoder(torch.cat([code_q, *code_outputs], axis=1)))

            code_outputs.append(code_q)
            upscale_counts.append(0)

        return decoder_outputs[-1]


from einops import pack


def equal(x, y):
    # return torch.all(torch.eq(x, y))
    return (((x - y).abs() < 1e-5).float()).mean()


if __name__ == "__main__":
    data = torch.randn(10, 30, 40)
    # convert float to torch.Tensor
    embed = nn.Embedding(10, 40)
    cx = torch.sum(data)
    print(cx)
    print(embed.weight.shape)
    one_hot = F.one_hot(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]), num_classes=10)
    x = torch.randn(40, 80)
    y = torch.randn(70, 80)
    dist = torch.cdist(x, y, p=2.0)
    min_encodings = torch.argmin(dist, dim=1)
    print(min_encodings)
    min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)  # (encoded_feat size,1)
    min_encodings = torch.zeros(40, 70)  # (encoded_feat size,embedding_size)
    min_encodings.scatter_(1, min_encoding_indices, 1)  # one-hot  like
    print(min_encodings.shape)
    my_min_encodings = F.one_hot(min_encoding_indices.squeeze(), num_classes=70)
    print(my_min_encodings.shape)
    print(equal(min_encodings, my_min_encodings))

    # dist_2 = torch.sqrt(
    #     torch.sum(x ** 2, dim=1, keepdim=True) +
    #     torch.sum(y ** 2, dim=1, keepdim=True).t() -
    #     2 * x @ y.t()
    # )
    # print(dist_2.shape)
    #
    # print(equal(dist, dist_2))
    #
    # x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # y = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]])
    #
    # # 使用 torch.cdist 计算欧氏距离
    # dist_cdist = torch.cdist(x, y, p=2.0)
    # # 手动计算欧氏距离
    # dist_manual = torch.sqrt(
    #     torch.sum(x ** 2, dim=1, keepdim=True) +
    #     torch.sum(y ** 2, dim=1, keepdim=True).t() -
    #     2 * x @ y.t()
    # )
    #
    # print("torch.cdist result:")
    # print(dist_cdist)
    # print("\nManual calculation result:")
    # print(dist_manual)
    # print(equal(dist_cdist, dist_manual))

    # top_k_values, top_k_indices = torch.topk(weights, 10)
    # print(top_k_values.shape, top_k_indices.shape)
    # top_k_mask = torch.zeros_like(weights)
    # top_k_mask[:, top_k_indices] = 1  # [L,C]
    #
    # print((top_k_mask * weights).shape)

    # x = torch.randn([10, 2])
    # y = torch.randn([5, 2])
    # c = x * y
    # print(c.shape)
    # x = torch.rand(528, 32)
    # y = torch.rand(2048, 32)
    # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    # cos_d = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
    # print(cos_d.shape)
    # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    # row_ind, col_ind = linear_sum_assignment(cost)
    # print(row_ind, col_ind)

    # attn = F.gumbel_softmax(logits, tau=1, dim=-1, hard=True)
    # codebook_indices = attn.argmax(dim=-1)
    # print(attn.shape, codebook_indices.shape)
    # feat_1 = torch.randn(10, 20)
    # feat_2 = torch.randn(15, 20)
    # distance = torch.cdist(feat_1, feat_2, p=2)
    # print(distance.shape)
    # distance_2 = (
    #         feat_1.pow(2).sum(1, keepdim=True)
    #         - 2 * feat_1 @ feat_2.t()
    #         + feat_2.pow(2).sum(1, keepdim=True).t()
    # )
    # print(distance_2.shape)
    # print((distance == distance_2).sum())
    # cos_d = F.cosine_similarity(feat_1.unsqueeze(1), feat_2.unsqueeze(0), dim=2)
    # print(cos_d.shape)
    # print(sum(cos_d))
    # embedding = nn.Embedding(10, 20)
    # print(embedding.weight.shape)
    # x.t().view(1, -1)  # Flattens the Tensor
    # model = nn.Conv2d(40, 10, 3, 1)
    # optimizer = torch.optim.AdamW(model.parameters())
    # for p in optimizer.param_groups:
    #     for k, v in p.items():
    #         print(k, "->", v)
    # f = cdll.LoadLibrary("./func.so")
    # print(f.func(99))

    # input = torch.randn(1, 20, 10)
    # model = nn.Linear(10, 3)
    # inter = model(input)
    # # model2 = nn.Linear(3, 1)
    # # output = model2(inter.detach())
    # loss = torch.mean(inter - 1)
    # loss.backward()
    # print(input.grad_fn, inter.grad)
    # pixel_percentage = {
    #     "background": 51.148658,
    #     "algae": 0.064494,
    #     "dead_twigs_leaves": 0.012825,
    #     "garbage": 2.345101,
    #     "water": 46.428922
    # }
    # # 计算初始权重
    # initial_weights = {k: 1 / v for k, v in pixel_percentage.items()}
    # weights = [initial_weights["background"], initial_weights["algae"], initial_weights["dead_twigs_leaves"],
    #            initial_weights["garbage"], initial_weights["water"]]
    # weights = torch.tensor(weights, dtype=torch.float32)
    # # 标准化权重，使其和等于类别数
    # class_weights = weights / weights.sum() * len(weights)
    # print(class_weights)
    # soft_weights = weights.softmax(dim=-1)
    # print(soft_weights)
    # start_x = 20
    # end_x = 100
    # start_y = 10
    # end_y = 70
    # pred_mask = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    # pred_mask[:, start_y:end_y, start_x:end_x] = 3
    # pred_mask[:, 80:90, 1:10] = 3
    # binary_mask = np.where(pred_mask == 3, 1, 0)
    # binary_mask = binary_mask.astype(np.uint8) * 255  # [B,H,W]
    # binary_mask = np.transpose(binary_mask, (1, 2, 0))  # [H,W,B]
    # # 膨胀操作
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # binary_mask = cv2.dilate(binary_mask, kernel2, iterations=2)
    # # 连通域分析
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    # # 遍历连通域分析结果，获取每个区域的掩码
    # # 没有连通的当作新的物体
    # masks = []
    # num_labels = min(num_labels, 3)
    # # 获取连通组件的面积
    # areas = stats[:, cv2.CC_STAT_AREA]
    # # 根据面积大小进行排序
    # sorted_indices = np.argsort(areas)[::-1][:num_labels]  # 降序排列的索引
    # region_mask = np.zeros((outputs.shape[2], outputs.shape[3], 3), np.uint8)
    # for index, label in enumerate(sorted_indices):
    #     if label == 0:
    #         continue
    #     # 获得每个不连通的物体
    #     # region_mask = np.uint8(labels == label) * 255  # 获取当前标签对应的区域掩码
    #     mask = labels == label
    #     region_mask[:, :, 0][mask] = 255
    #     region_stats = stats[label]  # 获取当前标签对应的区域统计信息
    #     masks.append((region_mask.copy(), region_stats))
    # # 显示每个区域的掩码图像
    # for region_idx, (region_mask, region_stats) in enumerate(masks):
    #     print(f"物体区域 {region_idx + 1}：")
    #     print("位置信息：", region_stats[:2])  # 左上角坐标
    #     print("形状信息：", region_stats[2:4])  # 宽度和高度
    #     cv2.imshow(f"Region {region_idx + 1}", region_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
