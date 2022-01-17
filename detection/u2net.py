import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, dirate: int = 1):
        super(ConvReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlockU(nn.Module):
    def __init__(self, in_channels: int = 3, midden_channels: int = 12, out_channels: int = 3):
        super(ResBlockU, self).__init__()
        self.conv_in = ConvReLU(in_channels, out_channels)

        self.convs = nn.ModuleList(
            [
                ConvReLU(out_channels, midden_channels),
                ConvReLU(midden_channels, midden_channels),
                ConvReLU(midden_channels, midden_channels),
                ConvReLU(midden_channels, midden_channels, 2),
            ]
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.decoder_conv3 = ConvReLU(midden_channels * 2, midden_channels)
        self.decoder_conv2 = ConvReLU(midden_channels * 2, midden_channels)
        self.decoder_conv1 = ConvReLU(midden_channels * 2, out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv_in(x)
        identity = x

        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            features.append(x)
            if i < 2:
                x = self.pool(x)

        features = features[::-1]
        x = self.decoder_conv3(torch.cat(features[0:2], 1))
        x = self.upsample(x)

        x = self.decoder_conv2(torch.cat((x, features[2]), 1))
        x = self.upsample(x)

        x = self.decoder_conv1(torch.cat((x, features[3]), 1)) + identity
        return x


class ResBlockUF(nn.Module):
    def __init__(self, in_channels: int = 3, midden_channels: int = 12, out_channels: int = 3):
        super(ResBlockUF, self).__init__()
        self.conv_in = ConvReLU(in_channels, out_channels)

        self.convs = nn.ModuleList(
            [
                ConvReLU(out_channels, midden_channels),
                ConvReLU(midden_channels, midden_channels, 2),
                ConvReLU(midden_channels, midden_channels, 4),
                ConvReLU(midden_channels, midden_channels, 8),
            ]
        )

        self.decoder_conv3 = ConvReLU(midden_channels * 2, midden_channels, 4)
        self.decoder_conv2 = ConvReLU(midden_channels * 2, midden_channels, 2)
        self.decoder_conv1 = ConvReLU(midden_channels * 2, out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv_in(x)
        identity = x

        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        features = features[::-1]

        x = self.decoder_conv3(torch.cat(features[0:2], 1))
        x = self.decoder_conv2(torch.cat((x, features[2]), 1))
        x = self.decoder_conv1(torch.cat((x, features[3]), 1)) + identity
        return x


class U2Net(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(U2Net, self).__init__()

        # encoder
        self.stages = nn.ModuleList(
            [
                ResBlockU(in_channels, 32, 64),
                ResBlockU(64, 32, 128),
                ResBlockU(128, 64, 256),
                ResBlockU(256, 128, 512),
                ResBlockUF(512, 256, 512)
            ]
        )
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage_final = ResBlockUF(512, 256, 512)

        # decoder
        self.stageds = nn.ModuleList(
            [
                ResBlockUF(1024, 256, 512),
                ResBlockU(1024, 128, 256),
                ResBlockU(512, 64, 128),
                ResBlockU(256, 32, 64),
                ResBlockU(128, 16, 64)
            ]
        )

        self.sides = nn.ModuleList(
            [
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Conv2d(64, 1, 3, padding=1),
                nn.Conv2d(128, 1, 3, padding=1),
                nn.Conv2d(256, 1, 3, padding=1),
                nn.Conv2d(512, 1, 3, padding=1),
                nn.Conv2d(512, 1, 3, padding=1)
            ]
        )

        self.upscores = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Upsample(scale_factor=4, mode='bilinear'),
                nn.Upsample(scale_factor=8, mode='bilinear'),
                nn.Upsample(scale_factor=16, mode='bilinear'),
                nn.Upsample(scale_factor=32, mode='bilinear')
            ]
        )
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x: torch.Tensor):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            x = self.pool(x)
        features = features[::-1]

        x = self.stage_final(x)
        x_down = x
        x = self.upscore2(x)

        features_up = []
        for i, staged in enumerate(self.stageds):
            x = staged(torch.cat((x, features[i]), 1))
            features_up.append(x)
            if i < 4:
                x = self.upscore2(x)
        features_up = features_up[::-1]
        features_up.append(x_down)

        out = []
        for i, (side, upscore) in enumerate(zip(self.sides, self.upscores)):
            x = side(features_up[i])
            if i == 0:
                out.append(x)
                continue
            x = upscore(x)
            out.append(x)

        result = self.outconv(torch.cat(out, 1))

        return result, out 


if __name__ == "__main__":
    from tensorboardX import SummaryWriter

    writer = SummaryWriter("log")
    model = U2Net().cuda()
    dummy_input = torch.rand(2, 3, 256, 256).cuda()
    print(model(dummy_input).shape)

    with SummaryWriter(comment="U2Net") as w:
        w.add_graph(model, (dummy_input,))
