import torch

from torch import nn


def simple_conv_block(in_channels, 
                      out_channels, 
                      kernel_size, 
                      stride, 
                      padding,
                      pool_size, 
                      pool_stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(pool_size, pool_stride))


class TwinNetwork(nn.Module):
    '''Feature extractor'''
    def __init__(self, output_dim=1024):
        super(TwinNetwork, self).__init__()
        self.output_dim = output_dim
        self.cnn1 = simple_conv_block(3, 32, 10, 1, 1, 2, 2)
        self.cnn2 = simple_conv_block(32, 64, 7, 1, 1, 2, 2)
        self.cnn3 = simple_conv_block(64, 128, 5, 1, 1, 2, 2)
        self.cnn4 = simple_conv_block(128, 256, 3, 1, 1, 2, 2)
        self.cnn5 = simple_conv_block(256, 512, 3, 1, 1, 2, 2)
        self.fc = nn.Sequential(
          nn.Linear(512*7*7, self.output_dim),
          nn.ReLU())
        
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = x.view(x.size()[0], -1)
        output = self.fc(x)
        return output


class SiameseNetwork(nn.Module):
    def __init__(self, twin_net):
        super(SiameseNetwork, self).__init__()
        self.twin_net = twin_net
        # PairwiseDistance
        self.fc = nn.Sequential(
            nn.Linear(self.twin_net.output_dim, 1),
            nn.Sigmoid())
        
    def forward(self, x_l, x_r):
        x_l = self.twin_net(x_l)
        x_r = self.twin_net(x_r)
        # calc dist
        x = torch.abs(torch.sub(x_l, x_r))
        output = self.fc(x)
        return output