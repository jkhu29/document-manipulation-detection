from matplotlib.pyplot import sca
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Optional

from .backbone import conv1x1, conv3x3, ChannelAttention


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        scale: int = 4,
        se: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.width = width
        self.scale = scale

        self.conv1 = conv1x1(inplanes, width * scale)
        self.bn1 = norm_layer(width * scale)
        self.conv2 = conv3x3(width * scale, width * scale, stride, groups, dilation)
        self.bn2 = norm_layer(width * scale)

        if scale == 1:
            self.num = 1
        else:
            self.num = scale - 1

        convs = []
        bns = []
        for _ in range(self.num):
            convs.append(conv3x3(width, width))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = ChannelAttention(channels=planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat([out, sp], dim=1)

        if self.scale != 1:
            out = torch.cat([out, spx[self.num]], dim=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
