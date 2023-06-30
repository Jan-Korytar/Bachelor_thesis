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


