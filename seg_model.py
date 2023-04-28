import torch
from torch import nn as nn
from torch.nn import functional as F


class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModel, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.upconv2 = nn.ConvTranspose2d(16, out_channels, kernel_size=5, stride=2)

    def forward(self, x):
        x_size = x.size()
        # Encoder
        x = F.relu(self.pool(self.bn1(self.conv1(x))))
        x = F.relu(self.pool(self.bn2(self.conv2(x))))

        x = F.relu(self.pool(self.bn3(self.upconv1(x))))
        x = F.relu(self.pool(self.upconv2(x)))

        x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)

        return torch.squeeze(x)
