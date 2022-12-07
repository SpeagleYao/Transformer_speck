import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + self.shortcut(x)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv1d(4, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.layer1 = self._make_layer(block, 32, num_blocks[0])
        self.layer2 = self._make_layer(block, 64, num_blocks[0])
        self.layer3 = self._make_layer(block, 128, num_blocks[0])
        self.layer4 = self._make_layer(block, 256, num_blocks[0])

        self.dense1 = self._dense_layer(4096, 4096)
        self.dense2 = self._dense_layer(4096, 1000)
        self.linear = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _dense_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.BatchNorm1d(out_dim), 
            nn.ReLU()
        )

    def forward(self, x):

        x = x.view(x.shape[0], 4, 16)
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        feature = self.flatten(out)
        out = self.dense1(feature)
        out = self.dense2(out)
        out = self.sigmoid(self.linear(out))
        return out

def ResNet18_my():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34_my():
    return ResNet(BasicBlock, [3, 4, 6, 3])

if __name__=='__main__':
    net = ResNet34_my()
    summary(net.cuda(), (1, 64))
    x = torch.rand(13, 64).cuda()
    y = net(x) 
    print(y.size())