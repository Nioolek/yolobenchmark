import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmyolo.registry import MODELS


from ..layers.ppyoloe_brick import EffectiveSELayer, ConvBNLayer, RepVggBlock


class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca'):
        super(CSPResStage, self).__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


@MODELS.register_module()
class PPYOLOEBackbone(nn.Module):

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3, 4],
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]

        if use_large_stem:
            self.stem = nn.Sequential(
                ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act),
                ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act),
                ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act)
            )
        else:
            self.stem = nn.Sequential(
                ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act),
                ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act)
            )
        n = len(channels) - 1
        self.stages = nn.Sequential(
            *[CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act) for i in range(n)]
        )
        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = inputs
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs


