import math

import torch
import torchvision.models as models
from torch import nn
from torch import optim


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


class Alexnet_finetune(nn.Module):
    def __init__(self, num_classes=31):
        super(Alexnet_finetune, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.final_classifier(x)
        return output


class DCCNet(nn.Module):
    def __init__(self, num_classes=31):
        super(DCCNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )

    def forward(self, source, target):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        source = self.bottleneck(source)

        mmd_loss = 0
        if self.training:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            target = self.bottleneck(target)
            mmd_loss += mmd_linear(source, target)

        result = self.final_classifier(source)

        return result, mmd_loss


def load_pretrained_alexnet(model):
    alexnet = models.alexnet(pretrained=True)
    pretrained_dict = alexnet.state_dict()
    model_dict = model.state_dict()
    for key, value in model_dict.item():
        if key.split('.')[0] in ['features', 'classifier']:
            model_dict[key] = pretrained_dict[key]
    model.load_state_dict(model_dict)
    return model


def step_decay(epoch, lr):
    initial_lr = lr
    drop = .8
    epochs_drop = 10.0
    lrate = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


BATCH_SIZE = 256
TRAIN_EPOCHS = 200
learning_rate = 1e-2
L2_DECAY = 5e-4
MOMENTUM = 0.9
cuda = torch.cuda.is_available()


def train_alexnet(epoch, model, lr, source_loader):
    log_interval = 10
    LEARNING_RATE = step_decay(epoch, lr)
    print(f'Learning Rate:{lr}')
    optimizer = optim.SGD([
        {'params': model.features.parameters()},
        {'params': model.classifier.parameters()},
        {'params': model.final_classifer.parameters(), 'lr': lr}
    ], lr=lr / 10, momentum=MOMENTUM, weight_decay=L2_DECAY)
    model.train()
    iter_source = iter(source_loader)
    num_iter = len(source_loader)
    correct = 0
    total_loss = 0
    clf_criterion = nn.CrossEntropyLoss()
    for i in range(1, num_iter):
        source_data, source_label = iter_source.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        optimizer.zero_grad()
        source_preds = model(source_data)
        preds = source_preds.data.max(1, keepdim=True)[1]
        correct += preds.eq(source_label.data.view_as(preds)).sum()
        loss = clf_criterion(source_preds, source_label)
        total_loss += loss
        loss.backward()
        optimizer.step()
        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(source_data), len(source_loader) * BATCH_SIZE,
                       100. * i / len(source_loader), loss.data[0]))

    total_loss /= len(source_loader)
    acc_train = float(correct) * 100. / (len(source_loader) * BATCH_SIZE)


def test_alexnet(model, target_loader):
    clf_criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in target_loader:
        if cuda:
            data, target = data.cuda(), target()
        target_preds = model(data)
        test_loss += clf_criterion(target_preds, target)  # sum up batch loss
        pred = target_preds.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(target_loader)
    return correct
