import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.transforms import v2 as transforms

class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels, base_dim=64, batch_norm=True, ):
        super(SegmentationModel, self).__init__()

        # Encoder
        base = base_dim
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(base) if batch_norm else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(base, 2*base, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(2*base) if batch_norm else nn.Identity()
        self.conv0 = nn.Conv2d(2*base, 4*base, kernel_size=3, padding=2)

        # decoder
        self.upconv0 = nn.ConvTranspose2d(4*base, 2*base, kernel_size=3, padding=2)

        self.upconv1 = nn.ConvTranspose2d(4*base, 1*base, kernel_size=3, padding=2)
        self.upconv_pool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn3 = nn.BatchNorm2d(base) if batch_norm else nn.Identity()
        self.upconv2 = nn.ConvTranspose2d(2*base, int( base/2), kernel_size=5, padding=2)
        self.upconv_pool2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3 = nn.Conv2d(int(base/2), out_channels,  kernel_size=1)

        self.skip_conv1 = nn.Conv2d(base, base, kernel_size=1, )
        self.skip_conv2 = nn.Conv2d(2 * base, int(2 * base), kernel_size=1)

    def forward(self, x): 
        x_size = x.size()
        # Encoder
        x1 = F.relu(self.pool(self.bn1(self.conv1(x))))
        x2 = F.relu(self.pool(self.bn2(self.conv2(x1))))
        x = F.relu(self.conv0(x2))
        x = F.relu(self.upconv0(x))
        x = torch.cat((x, self.skip_conv2(x2)), dim=1)
        x = F.relu(self.bn3(self.upconv1(self.upconv_pool1(x))))
        x = torch.cat((x, self.skip_conv1(x1)), dim=1)
        x = F.relu(self.upconv2(self.upconv_pool2(x)))
        x = self.conv3(x)
        x = F.softmax(x, dim=1)

        return x
