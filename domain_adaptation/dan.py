import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 50, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv2_drop = nn.Dropout()
        self.fc1 = nn.Linear(50 * 4 * 4, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn4 = nn.BatchNorm1d(100)

    def forward(self, input):
        x = F.max_pool2d(F.relu((self.bn1(self.conv1(input)))), 2)
        x = F.max_pool2d(F.relu((self.conv2_drop(self.bn2(self.conv2(x))))), 2)
        x = x.view(-1, 50 * 4 * 4)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.bn4(x)

        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        logits = self.fc3(input)
        return logits


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    return cost


def mmd_loss(source_features, target_features):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if use_gpu:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=torch.cuda.FloatTensor(sigmas)
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas=torch.FloatTensor(sigmas)
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
    loss_value = loss_value

    return loss_value


batch_size = 512
use_gpu = True
data_root = './data'
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)
mnist_path = data_root + '/MNIST'
mnistm_path = data_root + '/MNIST_M'
epochs = 1
plot_iter = 10
lr = 0.01
momentum = 0.9

theta1 = 0.5
theta2 = 0.5


def train(common_net, src_net, tgt_net, optimizer, criterion, epoch,
          source_dataloader, target_dataloader, train_hist):
    common_net.train()
    src_net.train()
    tgt_net.train()

    start_steps = epoch * len(source_dataloader)
    total_steps = epochs * len(source_dataloader)

    source_iter = iter(source_dataloader)
    target_iter = iter(target_dataloader)

    for batch_idx in range(min(len(source_dataloader), len(target_dataloader))):
        # get data
        sdata = next(source_iter)
        tdata = next(target_iter)

        # prepare the data
        input1, label1 = sdata
        input2, label2 = tdata
        if use_gpu:
            input1, label1 = input1.cuda(), label1.cuda()
            input2, label2 = input2.cuda(), label2.cuda()
        else:
            input1, label1 = input1, label1
            input2, label2 = input2, label2

        optimizer.zero_grad()

        input1 = input1.expand(input1.shape[0], 3, 28, 28)
        input = torch.cat((input1, input2), 0)
        common_feature = common_net(input)

        src_feature, tgt_feature = torch.split(common_feature, int(batch_size))

        src_output = src_net(src_feature)
        tgt_output = tgt_net(tgt_feature)

        class_loss = criterion(src_output, label1)

        mmd_loss = mmd_loss(src_feature, tgt_feature) * theta1 + \
                   mmd_loss(src_output, tgt_output) * theta2

        loss = class_loss + mmd_loss
        loss.backward()
        optimizer.step()
        step = epoch * len(target_dataloader) + batch_idx

        if (batch_idx + 1) % plot_iter == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tMMD Loss: {:.6f}'.format(
                batch_idx * len(input2), len(target_dataloader.dataset),
                100. * batch_idx / len(target_dataloader), loss.data[0], class_loss.data[0],
                mmd_loss.data[0]
            ))
            train_hist['Total_loss'].append(loss.cpu().data[0])
            train_hist['Class_loss'].append(class_loss.cpu().data[0])
