import torch
import torch.nn as nn
import numpy as np


class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


# x = torch.randn(10, 10)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # load
# checkpoint = torch.load("model.pth", map_location=device)
# m = myModel().to(device)
# m.load_state_dict(checkpoint, strict=True)
# print(m.fc)
# for key in checkpoint.keys():
#     print(key)
# save
# torch.save(m.state_dict(), "model.pth")


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)])
    )
    print(indices)
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


if __name__ == "__main__":
    # H = 10
    # W = 20
    # c = torch.randn(10, H * W)
    # k = int(H * W * 0.2)
    # _, indices = torch.topk(c, k, sorted=False)
    # h_coor = indices // W
    # print(h_coor)
    # w_coor = indices - h_coor * W
    # h_coor, w_coor = h_coor / H, w_coor / W
    # print(h_coor.shape, w_coor.shape)
    # pos_feat = torch.stack([h_coor, w_coor], dim=-1)
    # print(pos_feat.shape)
    a = [] + [2] + [532, 5]
    print(a)
