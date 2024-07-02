# Moment Matching for Multi-Source Domain Adaptation
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.autograd import Function
import torch.optim as optim
import torch.nn.functional as F


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class Identity(nn.Module):
    def __init__(self, mode='p3'):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class feature_extractor(nn.Module):

    def __init__(self):
        super(feature_extractor, self).__init__()

        self.cnn = models.resnet101(pretrained=True)
        self.cnn.fc = Identity()

        self.fc = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        with torch.no_grad():
            feature = self.cnn(x)
        feature = feature.view(x.size(0), -1)
        feature = self.fc(feature)

        return feature


class predictor(nn.Module):
    def __init__(self):
        super(predictor, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 345)
        )

    def forward(self, feature, reverse=False):
        if reverse:
            feature = grad_reverse(feature)
        feature = self.fc(feature)
        return feature


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(src_dataloader, tgt_dataloader):
    loss_extractor = nn.CrossEntropyLoss()
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    epochs = 20
    extractor = feature_extractor().to(device)
    extractor_optim = optim.Adam(extractor.parameters(), lr=3e-4)
    for i in range(epochs):
        source_ac = {}
        for source in source_domain:
            source_clf[source]['c1'] = source_clf[source]['c1'].train()
            source_clf[source]['c2'] = source_clf[source]['c2'].train()
            source_ac[source] = defaultdict(int)
            record = {}
            for source in source_domain:
                record[source] = {}
                for i in range(1, 3):
                    record[source][str(i)] = 0
            mcd_loss = 0
            dis_loss = 0
        for batch_index, (src_batch, tgt_batch) in enumerate(zip(src_dataloader, tgt_dataloader)):
            src_len = len(src_batch)
            loss_cls = 0
            # train extractor and source clssifier
            for index, batch in enumerate(src_batch):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y = y.view(-1)

                feature = extractor(x)
                pred1 = source_clf[source_domain[index]]['c1'](feature)
                pred2 = source_clf[source_domain[index]]['c2'](feature)

                source_ac[source_domain[index]]['c1'] += torch.sum(torch.max(pred1, dim=1)[1] == y).item()
                source_ac[source_domain[index]]['c2'] += torch.sum(torch.max(pred2, dim=1)[1] == y).item()
                loss1 = loss_extractor(pred1, y)
                loss2 = loss_extractor(pred2, y)
                loss_cls += loss1 + loss2
                record[source_domain[index]]['1'] += loss1.item()
                record[source_domain[index]]['2'] += loss2.item()
                if batch_index % 10 == 0:
                    for source in source_domain:
                        print(source)
                        print('c1 : [%.8f]' % (source_ac[source]['c1'] / (batch_index + 1) / BATCH_SIZE))
                        print('c2 : [%.8f]' % (source_ac[source]['c2'] / (batch_index + 1) / BATCH_SIZE))
                        # weights[index] = max([source_ac[source]['c1'], source_ac[source]['c2']])
                    # print('\n')

                m1_loss = 0
                m2_loss = 0
                for k in range(1, 3):
                    for i_index, batch in enumerate(src_batch):
                        x, y = batch
                        x = x.to(device)
                        y = y.to(device)
                        y = y.view(-1)

                        tar_x, _ = tgt_batch
                        tar_x = tar_x.to(device)

                        src_feature = extractor(x)
                        tar_feature = extractor(tar_x)

                        e_src = torch.mean(src_feature ** k, dim=0)
                        e_tar = torch.mean(tar_feature ** k, dim=0)
                        m1_dist = e_src.dist(e_tar)
                        m1_loss += m1_dist
                        for j_index, other_batch in enumerate(src_batch[i_index + 1:]):
                            other_x, other_y = other_batch
                            other_x = other_x.to(device)
                            other_y = other_y.to(device)
                            other_y = other_y.view(-1)
                            other_feature = extractor(other_x)

                            e_other = torch.mean(other_feature ** k, dim=0)
                            m2_dist = e_src.dist(e_other)
                            m2_loss += m2_dist

                loss_m = (epochs - i) / epochs * (m1_loss / N + m2_loss / N / (N - 1) * 2) * 0.8
                mcd_loss += loss_m.item()

                loss = loss_cls + loss_m

                if batch_index % 10 == 0:
                    print('[%d]/[%d]' % (batch_index, min_))
                    print('class loss : [%.5f]' % (loss_cls))
                    print('msd loss : [%.5f]' % (loss_m))

                extractor_optim.zero_grad()
                for source in source_domain:
                    source_clf[source]['optim'].zero_grad()

                loss.backward()

                extractor_optim.step()

                for source in source_domain:
                    source_clf[source]['optim'].step()
                    source_clf[source]['optim'].zero_grad()

                extractor_optim.zero_grad()

                tar_x, _ = tgt_batch
                tar_x = tar_x.to(device)
                tar_feature = extractor(tar_x)
                loss = 0
                d_loss = 0
                c_loss = 0
                for index, batch in enumerate(src_batch):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    y = y.view(-1)

                    feature = extractor(x)

                    pred1 = source_clf[source_domain[index]]['c1'](feature)
                    pred2 = source_clf[source_domain[index]]['c2'](feature)

                    c_loss += loss_extractor(pred1, y) + loss_extractor(pred2, y)

                    pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
                    pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
                    combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1)) / 2

                    d_loss += loss_l1(pred_c1, pred_c2)

                    for index_2, o_batch in enumerate(src_batch[index + 1:]):
                        pred_2_c1 = source_clf[source_domain[index_2 + index]]['c1'](tar_feature)
                        pred_2_c2 = source_clf[source_domain[index_2 + index]]['c2'](tar_feature)
                        combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1)) / 2

                        d_loss += loss_l1(combine1, combine2) * 0.1

                # discrepency_loss = torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
                # discrepency_loss = loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1))

                # loss += clf_loss - discrepency_loss
                loss = c_loss - d_loss

                loss.backward()
                extractor_optim.zero_grad()
                for source in source_domain:
                    source_clf[source]['optim'].zero_grad()

                for source in source_domain:
                    source_clf[source]['optim'].step()

                for source in source_domain:
                    source_clf[source]['optim'].zero_grad()

                all_dis = 0
                for i in range(3):
                    discrepency_loss = 0
                    tar_feature = extractor(tar_x)

                    for index, _ in enumerate(src_batch):

                        pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
                        pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
                        combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1)) / 2

                        discrepency_loss += loss_l1(pred_c1, pred_c2)

                        for index2, _ in enumerate(src_batch[index + 1:]):
                            pred_2_c1 = source_clf[source_domain[index2 + index]]['c1'](tar_feature)
                            pred_2_c2 = source_clf[source_domain[index2 + index]]['c2'](tar_feature)
                            combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1)) / 2

                            discrepency_loss += loss_l1(combine1, combine2) * 0.1
                    # discrepency_loss += torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
                    # discrepency_loss += loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1))

                    all_dis += discrepency_loss.item()

                    extractor_optim.zero_grad()

                    for source in source_domain:
                        source_clf[source]['optim'].zero_grad()

                    discrepency_loss.backward()

                    extractor_optim.step()
                    extractor_optim.zero_grad()

                    for source in source_domain:
                        source_clf[source]['optim'].zero_grad()

                dis_loss += all_dis


if __name__ == '__main__':
    argument = sys.argv[1:]
    source_domain = argument[:-1]
    target_domain = argument[-1]

    N = len(source_domain)
    source_dataloader_list = []
    source_clf = {}
    source_loss = {}
    BATCH_SIZE = 10
    for source in source_domain:
        dataset = Dataset(source, source + '_train.csv')
        dataset = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        if len(dataset) < min_:
            min_ = len(dataset)
        source_dataloader_list.append(dataset)

        # c1 : for target
        # c2 : for source
        source_clf[source] = {}
        source_clf[source]['c1'] = predictor().to(device)
        source_clf[source]['c2'] = predictor().to(device)
        source_clf[source]['optim'] = optim.Adam(
            list(source_clf[source]['c1'].parameters()) + list(source_clf[source]['c2'].parameters()), lr=3e-4)

    for source in source_domain:
        source_loss[source] = {}
        for i in range(1, 3):
            source_loss[source][str(i)] = {}
            source_loss[source][str(i)]['loss'] = []
            source_loss[source][str(i)]['ac'] = []
    mcd_loss_plot = []
    dis_loss_plot = []
