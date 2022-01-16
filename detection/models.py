
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, dirate: int = 1):
        super(ConvReLU, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlockU(nn.Module):
    def __init__(self, in_channels: int = 3, midden_channels: int = 12, out_channels: int = 3):
        super(ResBlockU, self).__init__()
        self.conv_in = ConvReLU(in_channels, out_channels)

        self.conv1 = ConvReLU(out_channels, midden_channels)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2 = ConvReLU(midden_channels, midden_channels)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3 = ConvReLU(midden_channels, midden_channels)
        self.conv4 = ConvReLU(midden_channels, midden_channels, 2)

        self.decoder_conv3 = ConvReLU(midden_channels * 2, midden_channels)
        self.decoder_conv2 = ConvReLU(midden_channels * 2, midden_channels)
        self.decoder_conv1 = ConvReLU(midden_channels * 2, out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        xin = self.conv_in(x)

        x1 = self.conv1(xin)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)

        x4 = self.conv4(x3)

        x3d = self.decoder_conv3(torch.cat((x4, x3), 1))
        x3dup = self.upsample(x3d)

        x2d = self.decoder_conv2(torch.cat((x3dup, x2), 1))
        x2dup = self.upsample(x2d)

        x1d = self.decoder_conv1(torch.cat((x2dup, x1), 1)) + xin
        return x1d


class ResBlockUF(nn.Module):
    def __init__(self, in_channels: int = 3, midden_channels: int = 12, out_channels: int = 3):
        super(ResBlockUF, self).__init__()
        self.conv_in = ConvReLU(in_channels, out_channels)

        self.conv1 = ConvReLU(out_channels, midden_channels)
        self.conv2 = ConvReLU(midden_channels, midden_channels, 2)
        self.conv3 = ConvReLU(midden_channels, midden_channels, 4)
        self.conv4 = ConvReLU(midden_channels, midden_channels, 8)

        self.decoder_conv3 = ConvReLU(midden_channels * 2, midden_channels, 4)
        self.decoder_conv2 = ConvReLU(midden_channels * 2, midden_channels, 2)
        self.decoder_conv1 = ConvReLU(midden_channels * 2, out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = self.conv_in(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x3d = self.decoder_conv3(torch.cat((x4, x3), 1))
        x2d = self.decoder_conv2(torch.cat((x3d, x2), 1))
        x1d = self.decoder_conv1(torch.cat((x2d, x1), 1)) + x
        return x1d


class U2Net(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(U2Net, self).__init__()

        self.stage1 = ResBlockU(in_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = ResBlockU(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = ResBlockU(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = ResBlockU(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = ResBlockUF(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = ResBlockUF(512, 256, 512)

        # decoder
        self.stage5d = ResBlockUF(1024, 256, 512)
        self.stage4d = ResBlockU(1024, 128, 256)
        self.stage3d = ResBlockU(512, 64, 128)
        self.stage2d = ResBlockU(256, 32, 64)
        self.stage1d = ResBlockU(128, 16, 64)

        self.side1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side2 = nn.Conv2d(64, 1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)

        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.outconv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = self.upscore2(hx6)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = self.upscore2(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upscore2(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upscore2(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upscore2(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = self.upscore2(d2)

        d3 = self.side3(hx3d)
        d3 = self.upscore3(d3)

        d4 = self.side4(hx4d)
        d4 = self.upscore4(d4)

        d5 = self.side5(hx5d)
        d5 = self.upscore5(d5)

        d6 = self.side6(hx6)
        d6 = self.upscore6(d6)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), \
               torch.sigmoid(d5), torch.sigmoid(d6)


if __name__ == "__main__":
    model = U2Net().cuda()
    test = torch.rand(2, 3, 192, 192).cuda()
    result = model(test)
    print(result[0].shape)
