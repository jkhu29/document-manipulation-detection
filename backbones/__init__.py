from .backbone import ResNet
from .resnet import Bottleneck as ResNetBottleneck
from .res2net import Bottleneck as Res2NetBottleneck


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet([3, 4, 6, 3], block=ResNetBottleneck, **kwargs)
    return model


def res2net50(**kwargs):
    """Constructs a Res2Net-50 model.
    """
    model = ResNet([3, 4, 6, 3], block=Res2NetBottleneck, **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet([3, 4, 23, 3], block=ResNetBottleneck, **kwargs)
    return model


def res2net101(**kwargs):
    """Constructs a Res2Net-101 model.
    """
    model = ResNet([3, 4, 23, 3], block=Res2NetBottleneck, **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet([3, 8, 36, 3], block=ResNetBottleneck, **kwargs)
    return model


def res2net152(**kwargs):
    """Constructs a Res2Net-152 model.
    """
    model = ResNet([3, 8, 36, 3], block=Res2NetBottleneck, **kwargs)
    return model


def resnext50_32x4d(**kwargs):
    """Constructs a ResNeXt-50_32x4d model.
    """
    model = ResNet([3, 4, 6, 3], block=ResNetBottleneck, groups=32, width_per_group=4, **kwargs)
    return model


def res2next50_32x4d(**kwargs):
    """Constructs a Res2NeXt-50_32x4d model.
    """
    model = ResNet([3, 4, 6, 3], block=Res2NetBottleneck, groups=32, width_per_group=4, **kwargs)
    return model


def resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt-101_32x8d model.
    """
    model = ResNet([3, 4, 23, 3], block=ResNetBottleneck, groups=32, width_per_group=8, **kwargs)
    return model


def res2next101_32x8d(**kwargs):
    """Constructs a Res2NeXt-101_32x8d model.
    """
    model = ResNet([3, 4, 23, 3], block=Res2NetBottleneck, groups=32, width_per_group=8, **kwargs)
    return model


def se_resnet50(**kwargs):
    """Constructs a SE-ResNet-50 model.
    """
    model = ResNet([3, 4, 6, 3], block=ResNetBottleneck, se=True, **kwargs)
    return model


def se_res2net50(**kwargs):
    """Constructs a SE-Res2Net-50 model.
    """
    model = ResNet([3, 4, 6, 3], block=Res2NetBottleneck, se=True, **kwargs)
    return model


def se_resnet101(**kwargs):
    """Constructs a SE-ResNet-101 model.
    """
    model = ResNet([3, 4, 23, 3], block=ResNetBottleneck, se=True, **kwargs)
    return model


def se_res2net101(**kwargs):
    """Constructs a SE-Res2Net-101 model.
    """
    model = ResNet([3, 4, 23, 3], block=Res2NetBottleneck, se=True, **kwargs)
    return model


def se_resnext101_32x8d(**kwargs):
    """Constructs a SE-ResNet-101 model.
    """
    model = ResNet([3, 4, 23, 3], block=ResNetBottleneck, groups=32, width_per_group=8, se=True, **kwargs)
    return model


def se_res2next101_32x8d(**kwargs):
    """Constructs a SE-Res2Net-101 model.
    """
    model = ResNet([3, 4, 23, 3], block=Res2NetBottleneck, groups=32, width_per_group=8, se=True, **kwargs)
    return model
