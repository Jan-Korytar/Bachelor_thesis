import torch
from torch import nn as nn
from torch.nn import functional as F


class bbox_model(nn.Module):
    def __init__(self, in_channels, base_dim=32, dropout=0.25, batch_norm=False, kernel_size=3):
        super(bbox_model, self).__init__()

        self.dropout = nn.Dropout2d(p=dropout)
        self.dropout2 = nn.Dropout(p=.2)

        self.conv1 = nn.Conv2d(in_channels, base_dim, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(base_dim) if batch_norm else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(base_dim, 2 * base_dim, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(2 * base_dim) if batch_norm else nn.Identity()
        self.conv3 = nn.Conv2d(2 * base_dim, 4 * base_dim, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(4 * base_dim) if batch_norm else nn.Identity()
        self.conv4 = nn.Conv2d(4 * base_dim, 8 * base_dim, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(8 * base_dim) if batch_norm else nn.Identity()
        self.bbox = nn.Linear(int(25 * 8 * base_dim), 512)
        self.bbox2 = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor):
        x = self.pool(self.dropout(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(F.relu(self.bn4(self.conv4(x)))))
        x = torch.flatten(x, start_dim=1)
        x = F.sigmoid(self.dropout2(self.bbox(x)))
        x = F.sigmoid(self.bbox2(x))
        return x


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
