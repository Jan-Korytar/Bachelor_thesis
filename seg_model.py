import torch
from torch import nn as nn
from torch.nn import functional as F


class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModel, self).__init__()

        # Encoder
        base = 20
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(base)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(base, 2*base, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(2*base)
        self.conv3 = nn.Conv2d(2 * base, 4 * base, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(4 * base)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(4*base, 2*base, kernel_size=5, stride=3)
        self.bn4 = nn.BatchNorm2d(2*base)
        self.upconv2 = nn.ConvTranspose2d(2*base, base, kernel_size=5, stride=2)
        self.bn5 = nn.BatchNorm2d(base)
        self.upconv3 = nn.ConvTranspose2d(base, out_channels, kernel_size=3,)

    def forward(self, x):
        x_size = x.size()
        # Encoder
        x = F.relu(self.pool(self.bn1(self.conv1(x))))
        x = F.relu(self.pool(self.bn2(self.conv2(x))))
        x = F.relu(self.pool(self.bn3(self.conv3(x))))

        x = F.relu(self.bn4(self.upconv1(x)))
        x = F.relu(self.bn5(self.upconv2(x)))
        x = F.relu(self.upconv3(x))

        x = F.interpolate(x, x_size[2:], mode='bilinear')

        return torch.squeeze(x)
