# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.dist import get_world_size

from mmyolo.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOV5OptimizerConstructor:

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.optimizer_cfg = optim_wrapper_cfg['optimizer']
        self.base_batch_size = 64

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()

        weight_decay = optimizer_cfg.pop('weight_decay')

        if 'batch_size' in optimizer_cfg:
            batch_size = optimizer_cfg.pop('batch_size')

            # scale weight decay
            total_batch_size = get_world_size() * batch_size
            accumulate = max(round(self.base_batch_size / total_batch_size), 1)
            weight_decay *= total_batch_size * accumulate / self.base_batch_size
            print(f'=========== Scaled weight_decay = {weight_decay}=========')

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items()
                   if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(
                    v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        # The order is very important and can't be messed up
        optimizer_cfg['params'] = []
        optimizer_cfg['params'].append({'params': g[1]})  # bn
        optimizer_cfg['params'].append({  # conv
            'params': g[0],
            'weight_decay': weight_decay
        })
        optimizer_cfg['params'].append(({'params': g[2]}))  # bias

        del g

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        del self.optim_wrapper_cfg['optimizer']
        optim_wrapper = OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
