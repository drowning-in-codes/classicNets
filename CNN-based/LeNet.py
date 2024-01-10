import torch.nn as nn
import torch.nn.functional as F


# LeNet for MNIST
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self,x):
        out = F.relu(self.conv1(x)) # 3*32*32 -> 6*28*28
        out = F.max_pool2d(out,2) # 6*28*28 -> 6*14*14
        out = F.relu(self.conv2(out)) # 6*14*14 -> 16*10*10
        out = F.max_pool2d(out, 2) # 16*10*10 -> 16*5*5
        out = out.view(out.size(0), -1) # 16*5*5 -> 400

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

