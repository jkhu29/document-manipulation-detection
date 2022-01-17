import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn import functional as F

from .classifier import Classifier
from .decode_heads.dahead import DAHead
import sys
sys.path.append("..")
import backbones as models


class ResNeXTDAHead(nn.Module):
    def __init__(
        self,
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

        self.encode_pretrain = models.resnext101_32x8d()
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
