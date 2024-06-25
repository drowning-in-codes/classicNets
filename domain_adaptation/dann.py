from typing import Any

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx: Any, x, alpha) -> Any:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class dann(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            OrderedDict(
                {'f_conv1': nn.Conv2d(3, 64, kernel_size=5),
                 'f_bn1': nn.BatchNorm2d(64),
                 'f_pool1': nn.MaxPool2d(2),
                 'f_relu': nn.ReLU(),
                 'f_conv2': nn.Conv2d(64, 50, kernel_size=5),
                 'f_bn2': nn.BatchNorm2d(50),
                 'f_drop1': nn.Dropout(),
                 'f_pool2': nn.MaxPool2d(2),
                 'f_relu2': nn.ReLU()
                 },
            )
        )
        self.class_classifier = nn.Sequential(OrderedDict(
            {'c_f1': nn.Linear(50 * 4 * 4, 100),
             'c_bn1': nn.BatchNorm1d(100),
             'c_relu1': nn.ReLU(),
             'c_drop1': nn.Dropout(),
             'c_fc2': nn.Linear(100, 100),
             'c_bn2': nn.BatchNorm1d(100),
             'c_relu2': nn.ReLU(),
             'c_fc3': nn.Linear(100, 10),
             'c_softmax': nn.LogSoftmax()
             }
        ))
        self.domain_classifier = nn.Sequential(OrderedDict(
            {'d_fc1': nn.Linear(50 * 4 * 4, 100),
             'd_bn1': nn.BatchNorm1d(100),
             'd_relu1': nn.ReLU(),
             'd_fc2': nn.Linear(100, 2),
             'd_softmax': nn.LogSoftmax(dim=1)
             }
        ))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.reshape(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
