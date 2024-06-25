import logging
import os
import time

import onnxruntime
import torch
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
            nn.MaxPool2d(2)
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
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)

if __name__ == "__main__":
    # Load pretrained model weights
    import torch
    import torchvision

    dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
    # model = torchvision.models.alexnet(pretrained=True).cuda()

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    # input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    # output_names = ["output1"]
    # torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names)
    import onnx

    onnx_model = onnx.load("alexnet.onnx")
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    print(onnx_model.model_version)
    ort_session = onnxruntime.InferenceSession("alexnet.onnx", providers=["CPUExecutionProvider"])


    # Print a human readable representation of the graph

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
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
