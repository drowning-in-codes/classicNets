# Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
import torch
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

# 通用参数
dataset_root = "dataset"
batch_size = 256
num_workers = 4
device = "cuda"
lr = 1e-4
N = 1
num_epoch = 100
models_save = "models_trained"


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        super(LeNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = torch.mean(x, 1).view(x.size()[0], 1, x.size()[2], x.size()[3])
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        # print(x.size())
        x = x.view(x.size(0), 48 * 4 * 4)
        return x


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class ClassifierVgg(nn.Module):
    def __init__(self, num_classes=31):
        super(ClassifierVgg, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class LeNetClassifier(nn.Module):
    def __init__(self, prob=0.5):
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x


extractor = LeNetEncoder()
classifier1 = LeNetClassifier()
classifier2 = LeNetClassifier()


def train(src_data, tgt_data, extractor, classifier1, classifier2):
    extractor.train()
    classifier1.train()
    classifier2.train()
    extractor.to(device)
    classifier1.to(device)
    classifier2.to(device)

    data = enumerate(zip(src_data, tgt_data))
    opt_e = torch.optim.Adam(extractor.parameters(), lr=lr)
    opt_c1 = torch.optim.Adam(classifier1.parameters(), lr=lr)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=lr)
    # 跳出循环
    len_dataloader = min(len(src_data), len(tgt_data))
    for idx, ((src_img, src_labels), (tgt_img, _)) in data:
        if idx > len_dataloader:
            break
        src_img = src_img.to(device)
        tgt_img = tgt_img.to(device)
        src_labels = src_labels.to(device)
        '''
        STEP A
        '''
        opt_e.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        src_feat = extractor(src_img)
        preds_s1 = classifier1(src_feat)
        preds_s2 = classifier2(src_feat)

        loss_A = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels)
        loss_A.backward()

        opt_e.step()
        opt_c1.step()
        opt_c2.step()
        '''
        STEP B
        '''
        opt_e.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        src_feat = extractor(src_img)
        preds_s1 = classifier1(src_feat)
        preds_s2 = classifier2(src_feat)

        src_tgt = extractor(tgt_img)
        preds_t1 = classifier1(src_tgt)
        preds_t2 = classifier2(src_tgt)

        loss_B = criterion(preds_s1, src_labels) + criterion(preds_s2, src_labels) - discrepancy(preds_t1, preds_t2)
        loss_B.backward()

        opt_c1.step()
        opt_c2.step()

        opt_e.zero_grad()
        opt_c1.zero_grad()
        opt_c2.zero_grad()

        '''
        STEP C
        '''
        for i in range(N):
            feat_tgt = extractor(tgt_img)
            preds_t1 = classifier1(feat_tgt)
            preds_t2 = classifier1(feat_tgt)
            loss_C = discrepancy(preds_t1, preds_t2)
            loss_C.backward()
            opt_e.step()

            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

        if (idx + 1) % 10 == 0:
            print(
                "loss_A = {:.2f}, loss_B = {:.2f}, loss_C = {:.2f}".format(loss_A.item(), loss_B.item(), loss_C.item()))
    return extractor, classifier1, classifier2
