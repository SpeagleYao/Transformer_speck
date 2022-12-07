import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm1d(planes, eps=1e-3, momentum=0.99)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()

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

        self.layers = nn.ModuleList()
        for nb in num_blocks:
            self.layers.append(self._make_layer(block, 32, nb))

        self.dense1 = self._dense_layer(512, 64)
        self.dense2 = self._dense_layer(64, 64)
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, planes))
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

        for i in range(len(self.layers)):
            out = self.layers[i](out)
        
        feature = self.flatten(out)
        out = self.dense1(feature)
        out = self.dense2(out)
        out = self.sigmoid(self.linear(out))
        return out

def ResNet_Gohr(blocks=10):
    return ResNet(BasicBlock, [1] * blocks)

if __name__=='__main__':
    net = ResNet_Gohr(1)
    summary(net.cuda(), (1, 64))
    x = torch.rand(13, 64).cuda()
    y = net(x) 
    print(y.size())