import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.layer5(x)
        x2 = self.layer6(x)
        return x1, x2

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        # b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


class FCDiscriminator_Local(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator_Local, self).__init__()

        self.conv1 = nn.Conv2d(num_classes + 2048, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if weight is not None:
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


NUM_CLASSES = 19
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP / 20)
POWER = 0.9
Lambda_weight = 0.01


class CrossEntropy2d(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        sm = nn.Softmax2d()

        P = sm(predict)
        P = torch.clamp(P, min=1e-9, max=1 - (1e-9))

        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask].view(1, -1)
        predict = P[target_mask.view(N, 1, H, W).repeat(1, C, 1, 1)].view(C, -1)
        probs = torch.gather(predict, dim=0, index=target)
        log_p = probs.log()
        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:

            loss = batch_loss.sum()
        return loss


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
             (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda(gpu)
    criterion = CrossEntropy2d(NUM_CLASSES).cuda(gpu)
    return criterion(pred, label)


Lambda_adv = 0.001
Epsilon = 0.4
Lambda_local = 40


def train():
    parser = ArgumentParser()
    args = parser.parse_args()
    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)
    # Create Network
    model = Res_Deeplab(num_classes=args.num_classes)
    model.train()
    model_D = FCDiscriminator(num_classes=args.num_classes)
    model_D.train()
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    if args.source == 'GTA5':
        trainloader = DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                           crop_size=input_size_source,
                           scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                crop_size=input_size_target,
                                                scale=True, mirror=True, mean=IMG_MEAN,
                                                set=args.set),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        damping = (1 - i_iter / NUM_STEPS)

        # ======================================================================================
        # train G
        # ======================================================================================

        # Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False

        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = images_s.cuda(args.gpu)
        pred_source1, pred_source2 = model(images_s)
        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)

        # Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s, args.gpu) + loss_calc(pred_source2, labels_s, args.gpu))
        loss_seg.backward()

        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, _, _, _ = batch
        images_t = images_t.cuda(args.gpu)

        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, dim=1), F.softmax(pred_target2, dim=1))

        D_out = interp_target(model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

        # Adaptive Adversarial Loss
        if i_iter > PREHEAT_STEPS:
            loss_adv = weighted_bce_loss(D_out,
                                         torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda(
                                             args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                                torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda(args.gpu))

        loss_adv = loss_adv * Lambda_adv * damping
        loss_adv.backward()

        # Weight Discrepancy Loss
        W5 = None
        W6 = None
        if args.model == 'ResNet':

            for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)

        loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1)  # +1 is for a positive loss
        loss_weight = loss_weight * Lambda_weight * damping * 2
        loss_weight.backward()

        # ======================================================================================
        # train D
        # ======================================================================================

        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True

        # Train with Source
        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()

        D_out_s = interp_source(model_D(F.softmax(pred_source1 + pred_source2, dim=1)))

        loss_D_s = bce_loss(D_out_s,
                            torch.FloatTensor(D_out_s.data.size().fill_(source_label)).cuda(args.gpu))

        loss_D_s.backward()

        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()

        D_out_t = interp_target(model_D(F.softmax(pred_target1 + pred_target2, dim=1)))

        # Adaptive Adversarial Loss
        if (i_iter > PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t,
                                         torch.FloatTensor(D_out_t.data.size()).fill_(target_label).cuda(
                                             args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_D_t = bce_loss(D_out_t,
                                torch.FloatTensor(D_out_t.data.size()).fill_(target_label).cuda(args.gpu))

        loss_D_t.backward()

        optimizer.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}'.format(
                i_iter, args.num_steps, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))

        f_loss = open(os.path.join(args.snapshot_dir, 'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(
            loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))
        f_loss.close()

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
