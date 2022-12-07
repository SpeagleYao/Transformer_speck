import math
import torch
from torch import Tensor
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + self.shortcut(x)
        
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer_crypto(nn.Module):

    def __init__(self):
        super(Transformer_crypto, self).__init__()

        self.DEPTH = 1

        self.conv1 = nn.Conv1d(4, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)

        # self.LN = nn.LayerNorm(32)
        # self.pos_encoder = PositionalEncoding(d_model=32, dropout=0.0, max_len=16)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=64, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.flatten = nn.Flatten()
        self.dense1 = self._dense_layer(512, 256)
        self.dense2 = self._dense_layer(256, 64)
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        # self.tc = nn.ModuleList()
        # for i in range(self.DEPTH):
            # self.tc.append(self._trans_conv_layer())
        # self.res = self._make_layer(BasicBlock, 32, 32, 1)

    def _make_layer(self, block, inplanes, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _dense_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.BatchNorm1d(out_dim), 
            nn.ReLU()
        )

    def _trans_conv_layer(self):
        return nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=32, nhead=2, dim_feedforward=128, dropout=0.0, batch_first=True), num_layers=1),
            nn.Conv1d(16, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

    def forward(self, x):

        out = x.view(x.shape[0], 4, 16)
        # out = self.LN(out)
        out = F.relu(self.bn1(self.conv1(out)))
        # out = self.pos_encoder(out)
        out = self.encoder(out)
        out = F.relu(self.bn2(self.conv2(out)))

        # for i in range(self.DEPTH):
            # out = self.tc[i](out)

        feature = self.flatten(out)
        out = self.dense1(feature)
        out = self.dense2(out)
        out = self.sigmoid(self.linear(out))

        return out

def transcrypto():
    return Transformer_crypto()

if __name__=='__main__':
    net = transcrypto().cuda()
    summary(net, (1, 1, 64))
    # x = torch.rand(13, 64).cuda()
    # y = net(x) 
    # print(y.size())