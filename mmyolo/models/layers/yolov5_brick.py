# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import digit_version


class Conv2dBatchLeaky(nn.Module):
    """This convenience layer groups a 2D convolution, a batchnorm and a leaky
    ReLU. They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                bias=False),
            nn.BatchNorm2d(self.out_channels),  # , eps=1e-6, momentum=0.01),
            nn.LeakyReLU(self.leaky_slope, inplace=True))

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


# spp??????
class SpatialPyramidPooling(nn.Module):

    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(pool_size, 1, pool_size // 2)
            for pool_size in pool_sizes
        ])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class MakeNConv(nn.Module):

    def __init__(self, filters_list, in_filters, n):
        super(MakeNConv, self).__init__()
        if n == 3:
            m = nn.Sequential(
                Conv2dBatchLeaky(in_filters, filters_list[0], 1, 1),
                Conv2dBatchLeaky(filters_list[0], filters_list[1], 3, 1),
                Conv2dBatchLeaky(filters_list[1], filters_list[0], 1, 1),
            )
        elif n == 5:
            m = nn.Sequential(
                Conv2dBatchLeaky(in_filters, filters_list[0], 1, 1),
                Conv2dBatchLeaky(filters_list[0], filters_list[1], 3, 1),
                Conv2dBatchLeaky(filters_list[1], filters_list[0], 1, 1),
                Conv2dBatchLeaky(filters_list[0], filters_list[1], 3, 1),
                Conv2dBatchLeaky(filters_list[1], filters_list[0], 1, 1),
            )
        else:
            raise NotImplementedError
        self.m = m

    def forward(self, x):
        return self.m(x)


class Transition(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        half_nchannels = int(nchannels / 2)
        layers = [
            Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            nn.Upsample(scale_factor=2)
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return x


class FuseStage(nn.Module):
    custom_layers = (Transition, )

    def __init__(self, in_filter, is_reversal=False):
        super(FuseStage, self).__init__()
        if is_reversal:
            self.left_conv = Conv2dBatchLeaky(in_filter, in_filter * 2, 3, 2)
            self.right_conv = None
        else:
            self.right_conv = Transition(in_filter)
            self.left_conv = Conv2dBatchLeaky(in_filter, in_filter // 2, 1, 1)

    def forward(self, data):
        left, right = data
        left = self.left_conv(left)
        if self.right_conv:
            right = self.right_conv(right)
        return torch.cat((left, right), dim=1)


class StageBlock(nn.Module):
    custom_layers = ()

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),
            Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1))

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):
    custom_layers = (StageBlock, StageBlock.custom_layers)

    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(StageBlock(2 * nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


class HeadBody(nn.Module):
    custom_layers = ()

    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels / 2)
        else:
            half_nchannels = int(nchannels / 3)
        in_nchannels = 2 * half_nchannels
        layers = [
            Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
            Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x


class Head(nn.Module):

    def __init__(self, nchannels, nanchors, nclasses):
        super().__init__()
        mid_nchannels = 2 * nchannels
        layer_list = [
            Conv2dBatchLeaky(nchannels, mid_nchannels, 3, 1),
            nn.Conv2d(mid_nchannels, nanchors * (5 + nclasses), 1, 1, 0),
        ]
        self.feature = nn.Sequential(*layer_list)

    def forward(self, data):
        x = self.feature(data)
        return x


# yolov5??????
# ?????????????????????????????????mmdetection?????????????????????


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, self.inplace) / 6
        return out


class Identity(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class SiLU(nn.Module):

    def __init__(self, inplace=True):
        super(SiLU, self).__init__()

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # note: momentum and eps
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=0.001)
        if digit_version(torch.__version__) >= digit_version('1.7.0'):
            self.act = nn.SiLU(inplace=True) if act is True else (
                act if isinstance(act, nn.Module) else nn.Identity())
        else:
            self.act = SiLU(inplace=True) if act is True else (
                act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat([
                x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ], 1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self,
                 c1,
                 c2,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # BN?????????????????????????????????????????????????????????????????????????????????
        self.bn = nn.BatchNorm2d(
            2 * c_, eps=0.001, momentum=0.01)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self,
                 c1,
                 c2,
                 n=1,
                 shortcut=True,
                 g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
