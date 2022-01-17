from .common import * 


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
