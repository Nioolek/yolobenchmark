# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.dist import get_world_size

from mmyolo.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS
from mmengine.logging import print_log


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOV7OptimizerConstructor:

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.optimizer_cfg = optim_wrapper_cfg['optimizer']
        self.base_batch_size = 64

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        weight_decay = optimizer_cfg.pop('weight_decay')
        batch_size = optimizer_cfg.pop('batch_size')

        # scale weight decay
        total_batch_size = get_world_size() * batch_size
        accumulate = max(round(self.base_batch_size / total_batch_size), 1)
        weight_decay *= total_batch_size * accumulate / self.base_batch_size
        print(f'=========== Scaled weight_decay = {weight_decay}=========')

        bn = tuple(v for k, v in nn.__dict__.items()
                   if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, bn):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):
                    pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):
                    pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):
                    pg0.append(v.imo.implicit)
                else:
                    for iv in v.imo:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)
            if hasattr(v, 'attn'):
                if hasattr(v.attn, 'logit_scale'):
                    pg0.append(v.attn.logit_scale)
                if hasattr(v.attn, 'q_bias'):
                    pg0.append(v.attn.q_bias)
                if hasattr(v.attn, 'v_bias'):
                    pg0.append(v.attn.v_bias)
                if hasattr(v.attn, 'relative_position_bias_table'):
                    pg0.append(v.attn.relative_position_bias_table)
            if hasattr(v, 'rbr_dense'):
                if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                    pg0.append(v.rbr_dense.weight_rbr_origin)
                if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                    pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                    pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                    pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                    pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(v.rbr_dense, 'vector'):
                    pg0.append(v.rbr_dense.vector)

        # The order is very important and can't be messed up
        optimizer_cfg['params'] = []
        optimizer_cfg['params'].append({'params': pg0})  # bn
        optimizer_cfg['params'].append({  # conv
            'params': pg1,
            'weight_decay': weight_decay
        })
        optimizer_cfg['params'].append(({'params': pg2}))  # bias

        print_log('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)), 'current')

        del pg0, pg1, pg2

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        del self.optim_wrapper_cfg['optimizer']
        optim_wrapper = OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
