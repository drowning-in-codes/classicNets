import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import random


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, loss_coeff=1.0):
        super(DeepInfoMaxLoss, self).__init__()
        self.loss_coeff = loss_coeff

    def __call__(self, x, y):
        joint_expectation = (-F.softplus(-x)).mean()
        marginal_expectation = F.softplus(y).mean()
        mutual_info = joint_expectation - marginal_expectation
        return -mutual_info * self.loss_coeff


class StatisticsNetwork(nn.Module):
    def __init__(self, input_dim):
        super(StatisticsNetwork, self).__init__()
        hidden_dim = input_dim * 2
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.conv3 = nn.Conv2d(hidden_dim, input_dim, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        local_statistics = self.conv3(x)
        return local_statistics
