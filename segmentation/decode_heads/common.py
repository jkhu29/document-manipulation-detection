import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ConvReLU(nn.Module):
    """docstring for ConvReLU"""

    def __init__(
        self, 
        channels: int = 256, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        dilation: int = 1,
        out_channels=None
    ):
        super(ConvReLU, self).__init__()
        if out_channels is None:
            self.out_channels = channels
        else:
            self.out_channels = out_channels

        self.conv = nn.Conv2d(
            channels, self.out_channels, 
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        return x * score


class PositionAttention(nn.Module):
    """docstring for PositionAttention"""

    def __init__(self, channels: int = 256, reduction: int = 8):
        super(PositionAttention, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv2 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, _, h, w = x.shape
        x_embed1 = self.conv1(x).view(n, -1, h * w).permute(0, 2, 1)
        x_embed2 = self.conv2(x).view(n, -1, h * w)

        attention = F.softmax(torch.bmm(x_embed1, x_embed2), dim=-1)

        x_embed1 = self.conv3(x).view(n, -1, h * w)
        x_embed2 = torch.bmm(x_embed1, attention.permute(0, 2, 1)).view(n, -1, h, w)
        x = self.alpha * x_embed2 + x
        return x
