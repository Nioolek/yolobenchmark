# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
from mmengine.hooks import ParamSchedulerHook

from mmyolo.registry import HOOKS


def linear(lrf, epochs):
    return lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf


def cosine(lrf, epochs):
    return lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (lrf - 1) + 1


@HOOKS.register_module()
class YOLOV5LrUpdaterHook(ParamSchedulerHook):
    """YOLOX learning rate scheme.

    There are two main differences between YOLOXLrUpdaterHook
    and CosineAnnealingLrUpdaterHook.

       1. When the current running epoch is greater than
           `max_epoch-last_epoch`, a fixed learning rate will be used
       2. The exp warmup scheme is different with LrUpdaterHook in MMCV
    """
    priority = 9

    def __init__(self, repeat_num=1, max_epoch=300, lrf=0.01, lr_scheduler='linear'):
        super(YOLOV5LrUpdaterHook, self).__init__()
        # self.warmup_iters = 1000
        # self.xi = [0, self.warmup_iters]
        self.warmup_bias_lr = 0.1
        self.warmup_momentum = 0.8
        self.warmup_epochs = 3
        self.repeat_num = repeat_num
        self.momentum = 0.937
        self.lf = eval(lr_scheduler)(lrf, max_epoch)
        self.warmup_end = False

    def before_train(self, runner) -> None:
        optimizer = runner.optim_wrapper.optimizer
        for group in optimizer.param_groups:
            # If the param is never be scheduled, record the current value
            # as the initial value.
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch=None) -> None:
        cur_iters = runner.iter
        cur_epoch = runner.epoch
        optimizer = runner.optim_wrapper.optimizer

        # The minimum warmup is 1000
        dataloader_len = len(runner.train_loop.dataloader) // self.repeat_num
        warmup_total_iters = max(
            round(self.warmup_epochs * dataloader_len), 1000)
        xi = [0, warmup_total_iters]

        if cur_iters <= warmup_total_iters:
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(cur_iters, xi, [
                    self.warmup_bias_lr if j == 2 else 0.0,
                    self.base_lr[j] * self.lf(cur_epoch)
                ])
                if 'momentum' in x:
                    x['momentum'] = np.interp(
                        cur_iters, xi, [self.warmup_momentum, self.momentum])
            # print('xxxxx-ni=', cur_iters, [x['lr'] for x in optimizer.param_groups])
        else:
            self.warmup_end = True

    def after_train_epoch(self, runner):
        if self.warmup_end:
            cur_epoch = runner.epoch
            optimizer = runner.optim_wrapper.optimizer
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = self.base_lr[j] * self.lf(cur_epoch)
            # print('xxxxx-ni=', 1, [x['lr'] for x in optimizer.param_groups])
