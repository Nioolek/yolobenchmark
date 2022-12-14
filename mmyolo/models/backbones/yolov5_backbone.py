# Copyright (c) OpenMMLab. All rights reserved.
import functools
import math

import torch.nn as nn

from mmyolo.registry import MODELS
from ..layers import yolov5_brick as vn_layer


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def _make_round(x, depth_multiple=1.0):
    return max(round(x * depth_multiple), 1) if x > 1 else x


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(
        _make_divisible, divisor=divisor, width_multiple=width_multiple)


def make_round(depth_multiple=1.0):
    return functools.partial(_make_round, depth_multiple=depth_multiple)


@MODELS.register_module()
class YOLOV5Backbone(nn.Module):

    def __init__(self,
                 input_channel=3,
                 depth_multiple=1.0,
                 width_multiple=1.0):
        # yolov5s: depth_multiple=0.33,width_multiple=0.5
        super(YOLOV5Backbone, self).__init__()
        self.depth_multiple = depth_multiple  # layer方向
        self.width_multiple = width_multiple  # channel方向
        self.return_index = [4, 6, 9]
        make_div8_fun = make_divisible(8, self.width_multiple)
        make_round_fun = make_round(self.depth_multiple)

        model = []

        # focal = vn_layer.Focus(input_channel, make_div8_fun(64), k=3)
        focal = vn_layer.Conv(input_channel, make_div8_fun(64), k=6, s=2, p=2)
        model.append(focal)

        conv1 = vn_layer.Conv(make_div8_fun(64), make_div8_fun(128), k=3, s=2)
        model.append(conv1)
        bsp1 = vn_layer.C3(
            make_div8_fun(128), make_div8_fun(128), make_round_fun(3))
        model.append(bsp1)
        conv2 = vn_layer.Conv(make_div8_fun(128), make_div8_fun(256), k=3, s=2)
        model.append(conv2)

        # bsp2 = vn_layer.C3(make_div8_fun(256), make_div8_fun(256), make_round_fun(9))
        bsp2 = vn_layer.C3(
            make_div8_fun(256), make_div8_fun(256), make_round_fun(6))
        model.append(bsp2)  # 第一个输出层

        conv3 = vn_layer.Conv(make_div8_fun(256), make_div8_fun(512), k=3, s=2)
        model.append(conv3)
        bsp3 = vn_layer.C3(
            make_div8_fun(512), make_div8_fun(512), make_round_fun(9))
        model.append(bsp3)  # 第二个输出层

        conv4 = vn_layer.Conv(
            make_div8_fun(512), make_div8_fun(1024), k=3, s=2)
        model.append(conv4)

        # spp1 = vn_layer.SPP(make_div8_fun(1024), make_div8_fun(1024))
        # model.append(spp1)
        # bsp4 = vn_layer.C3(make_div8_fun(1024), make_div8_fun(1024), make_round_fun(3), shortcut=False)
        # model.append(bsp4)  # 第三个输出层

        bsp4 = vn_layer.C3(
            make_div8_fun(1024), make_div8_fun(1024), make_round_fun(3))
        model.append(bsp4)
        spp1 = vn_layer.SPPF(make_div8_fun(1024), make_div8_fun(1024))
        model.append(spp1)  # 第三个输出层
        self.backbone = nn.Sequential(*model)
        # self.init_weights()

    # for test
    # def init_weights(self):
    #     from mmcv.cnn import constant_init
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             constant_init(m, 1)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             constant_init(m, 1)

    def forward(self, x):
        # np.save('image.npy', x.detach().cpu().numpy())
        out = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            # print(m, x.sum(), x.mean())
            if i in self.return_index:
                out.append(x)
        return out  # 从大到小
