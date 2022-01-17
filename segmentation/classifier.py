import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class Classifier(nn.Module):
    def __init__(self, in_channels, channels, num_conv: int = 3, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv = nn.Sequential(
            self._dsconv(in_channels, channels),
            self._dsconv(channels, channels),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.conv_lists = nn.ModuleList([self.conv for _ in range(num_conv)])

        self.conv_fuse = nn.Conv2d(num_conv, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def _dsconv(self, in_channels, channels):
        return nn.Sequential(
            # depthwise conv
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            # pointwise conv
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(channels),
            # activate
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: Tensor):
        result = []
        for conv in self.conv_lists:
            x = conv(x)
            result.append(x)
        result = torch.cat(result, dim=1)
        result = self.conv_fuse(result)
        result = F.interpolate(result, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return result
