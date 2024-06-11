import torch
import torch.nn as nn


def create_2d_relative_bias_trainable_embeddings(n_head, height, width, dim):
    # width:5 [0 1 2 3 4]  bias:[-4,4] 2*width-1
    # height:5 [0 1 2 3 4] bias:[-4 ,4] 2*height- 1
    pos_embedding = nn.Embedding((2 * width - 1) * (2 * height - 1), n_head)
    nn.init.xavier_uniform_(pos_embedding.weight)

    def get_2d_relative_position(height, width):
        m1, m2 = torch.meshgrid(torch.arange(height), torch.arange(width))
        coords = torch.stack([m1, m2], 0)  # [2,h,w]
        # m1 00000
        #    11111
        # m2 01234
        #    01234
        # 0000011111
        # 0123401234
        coords_flatten = torch.flatten(coords, 1)  # [2,h*w]
        relative_coords_bias = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2,h*w,h*w]
        relative_coords_bias[0, :, :] += height - 1
        relative_coords_bias[1, :, :] += width - 1
        # A:2d B:1d B[i*cols+j] = A[i,j]
        relative_coords_bias[0, :, :] *= width
        return relative_coords_bias.sum(0)  # [h*w,h*w]

    relative_position_bias = get_2d_relative_position(height, width)

    relative_position_bias = torch.flatten(relative_position_bias)
    bias_embedding = pos_embedding(relative_position_bias).reshape(height * width, height * width, n_head)
    bias_embedding = bias_embedding.permute(2, 0, 1).unsqueeze(0)  # [1,n_head,height*width,height*width]
    return bias_embedding


if __name__ == '__main__':
    pos_embedding = create_2d_relative_bias_trainable_embeddings(3, 3, 3, 64)
