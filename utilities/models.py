import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class UNet_segmentation(torch.nn.Module):
    def __init__(self, depth, base_dim, in_channels, out_channels):
        """
        Initializes the UNet model.

        Parameters:
        - depth (int): Depth of the U-Net architecture.
        - base_dim (int): Number of channels in the first encoder block.
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        """
        super(UNet_segmentation, self).__init__()

        # Encoder
        self.depth = depth
        self.e1 = EncoderBlock(in_channels, base_dim)
        for i in range(2, depth + 1):
            setattr(self, f'e{i}', EncoderBlock(int(base_dim), int(base_dim * 2)))
            base_dim *= 2

        # Bottleneck
        self.b = ConvBlock(base_dim, base_dim * 2)
        base_dim *= 2

        # Decoder
        for i in range(1, depth + 1):
            setattr(self, f'd{i}', DecoderBlock(int(base_dim), int(base_dim / 2)))
            base_dim /= 2

        # Classifier
        self.outputs = nn.Conv2d(int(base_dim), out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        # Encoder
        s = [None] * self.depth
        p = [None] * self.depth
        s[0], p[0] = self.e1(inputs)
        for i in range(2, self.depth + 1):
            e_block = getattr(self, f'e{i}')
            s[i - 1], p[i - 1] = e_block(p[i - 2])

        # Bottleneck
        b = self.b(p[self.depth - 1])  # C, 1024, 4, 4 for 128**2 input

        # Decoder
        for i in range(1, self.depth + 1):
            d_block = getattr(self, f'd{i}')
            b = d_block(b, s[self.depth - i])

        # Classifier
        outputs = self.outputs(b)
        return outputs


class ConvBlockBBox(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)

        return x


class RegressorBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.fc1 = nn.Linear(in_c, out_c)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        return x


class BboxModel(nn.Module):
    def __init__(self, in_channels, base_dim=64, dropout=0.25, batch_norm=False, kernel_size=3, depth=6, img_dim=512):
        super(BboxModel, self).__init__()

        self.depth = depth
        self.base_dim_start = base_dim
        self.img_dim = img_dim
        self.e1 = ConvBlockBBox(in_channels, base_dim)
        for i in range(2, depth + 1):
            setattr(self, f'e{i}', ConvBlockBBox(int(base_dim), int(base_dim * 2)))
            base_dim *= 2

        self.r1 = RegressorBlock(int(((img_dim / (2 ** (depth))) ** 2) * (self.base_dim_start * 2 ** (depth - 1))),
                                 4096)  # 1024, 4, 4 for 3* 128**2 input
        self.r2 = RegressorBlock(4096, 2048)
        self.r3 = RegressorBlock(2048, 1024)
        self.r4 = nn.Linear(1024, 4)

    def forward(self, x: torch.Tensor):
        s = self.e1(x)
        for i in range(2, self.depth + 1):
            e_block = getattr(self, f'e{i}')
            s = e_block(s)

        b = s.view(x.shape[0], -1)

        x = self.r1(b)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        return x
