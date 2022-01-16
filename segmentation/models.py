import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn import functional as F

import torchvision.models as models


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


class DAHead(nn.Module):
    def __init__(
        self, 
        in_channels:int = 128,
        channels:int = 128,
        num_classes:int = 128,
        pam_channels:int = 128
    ):
        super().__init__()

        self.pam_in_conv = ConvReLU(in_channels, out_channels=pam_channels)
        self.pam = PositionAttention(pam_channels)
        self.pam_out_conv = nn.Sequential(
            nn.Conv2d(pam_channels, num_classes, 1),
        )

        self.cam_in_conv = ConvReLU(in_channels, out_channels=channels)
        self.cam = ChannelAttention(channels)
        self.cam_out_conv = nn.Sequential(
            nn.Conv2d(pam_channels, num_classes, 1),
        )

    def forward(self, x: Tensor):
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_feat = F.interpolate(pam_feat, scale_factor=2, mode="bilinear", align_corners=False)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_feat = F.interpolate(cam_feat, scale_factor=2, mode="bilinear", align_corners=False)

        return pam_feat + cam_feat


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


class ResNeXTDAHead(nn.Module):
    def __init__(
        self,
        pretrain: bool = False,
        in_channels: int = 3,
        img_size: int = 128,
        classifier_channels: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()

        print("model building...")
        self.in_channels = in_channels
        self.img_size = img_size
        self.classifier_channels = classifier_channels
        self.num_classes = num_classes

        self.encode_pretrain = models.resnext101_32x8d(pretrained=pretrain)
        self.encoder = nn.ModuleList(
            [
                self.encode_pretrain.layer1,
                self.encode_pretrain.layer2,
                self.encode_pretrain.layer3,
                self.encode_pretrain.layer4,
            ]
        )

    def _build_decoder_head(self, features):
        DAHeads = []
        for i, feature in enumerate(features):
            DAHeads.append(
                DAHead(
                    in_channels=feature.shape[1],
                    channels=feature.shape[1] // 2,
                    num_classes=feature.shape[1] // 2,
                    pam_channels=feature.shape[1] // 2
                ).to(feature.device)
            )
            if i == len(features) - 2:
                break
        self.decode_head = nn.ModuleList(DAHeads)

    def _build_decoder_classifier(self, decode_head_outs):
        self.decode_classifier = Classifier(
            in_channels=decode_head_outs[-1].shape[1],
            channels=self.classifier_channels,
            num_conv=1,
            scale_factor=self.img_size // decode_head_outs[-1].shape[2]
        ).to(decode_head_outs[0].device)

    def build_model(self, device="cuda"):
        x = torch.rand((2, 3, self.img_size, self.img_size)).to(device)
        x = self.encode_pretrain.conv1(x)
        x = self.encode_pretrain.bn1(x)
        x = self.encode_pretrain.relu(x)

        features = []
        for encode_layer in self.encoder:
            x = encode_layer(x)
            features.append(x)
        features = features[::-1]

        decode_head_outs = []
        self._build_decoder_head(features)
        for feature, decode_head_layer in zip(features, self.decode_head):
            decode_head_out = decode_head_layer(feature)
            decode_head_outs.append(decode_head_out)

        self._build_decoder_classifier(decode_head_outs)
        print("over")

    def forward(self, x: Tensor):
        x = self.encode_pretrain.conv1(x)
        x = self.encode_pretrain.bn1(x)
        x = self.encode_pretrain.relu(x)

        features = []
        for encode_layer in self.encoder:
            x = encode_layer(x)
            features.append(x)
        features = features[::-1]

        for i, decode_head_layer in enumerate(self.decode_head):
            feature = decode_head_layer(x)
            x = feature + features[i+1]

        x = self.decode_classifier(x)

        return torch.sigmoid(x)


if __name__ == "__main__":
    from tensorboardX import SummaryWriter

    writer = SummaryWriter("log")
    model = ResNeXTDAHead().cuda()
    model.build_model()
    dummy_input = torch.rand(2, 3, 256, 256).cuda()
    with SummaryWriter(comment="ResNeXTDAHeadUNet") as w:
        w.add_graph(model, (dummy_input,))
