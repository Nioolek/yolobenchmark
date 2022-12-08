# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
from mmengine.hooks import ParamSchedulerHook
from mmengine.model.wrappers import is_model_wrapper

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class PPYOLOELrUpdaterHook(ParamSchedulerHook):
    """YOLOX learning rate scheme.

    There are two main differences between YOLOXLrUpdaterHook
    and CosineAnnealingLrUpdaterHook.

       1. When the current running epoch is greater than
           `max_epoch-last_epoch`, a fixed learning rate will be used
       2. The exp warmup scheme is different with LrUpdaterHook in MMCV
    """
    priority = 9

    def __init__(self,
                 start_factor=0.,
                 warmup_epochs=5,
                 min_lr_ratio=0.0,
                 total_epochs=360,
                 repeat_num=1
                 ):
        super(PPYOLOELrUpdaterHook, self).__init__()
        self.start_factor = start_factor
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs

        # # self.warmup_iters = 1000
        # # self.xi = [0, self.warmup_iters]
        # self.warmup_bias_lr = 0.1
        # self.warmup_momentum = 0.8
        # self.warmup_epochs = 3
        self.repeat_num = repeat_num
        # self.momentum = 0.937
        # self.lf = eval(lr_scheduler)(lrf, max_epoch)
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
        self.min_lr = [i * self.min_lr_ratio for i in self.base_lr]


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
        # xi = [0, warmup_total_iters]
        total_iters = self.total_epochs * dataloader_len

        for j, x in enumerate(optimizer.param_groups):
            if cur_iters <= warmup_total_iters:
                alpha = cur_iters / warmup_total_iters
                factor = self.start_factor * (1 - alpha) + alpha
                lr = self.base_lr[j] * factor
            else:
                lr = self.min_lr[j] + (self.base_lr[j] - self.min_lr[j]) * 0.5 * (math.cos(
                    (cur_iters - warmup_total_iters) * math.pi /
                    (total_iters - warmup_total_iters)) + 1.0)
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = lr

@HOOKS.register_module()
class PPYOLOEParamSchedulerHook(ParamSchedulerHook):
    """A hook to update learning rate and momentum in optimizer of PPYOLOE.

    Args:
        warmup_min_iter (int): Minimum warmup iters. Defaults to 1000.
        start_factor (float): The number we multiply learning rate in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 0.
        warmup_epochs (int): Epochs for warmup. Defaults to 5.
        min_lr_ratio (float): Minimum learning rate ratio.
        total_epochs (int): In PPYOLOE, `total_epochs` is set to
            training_epochs x 1.2. Defaults to 360.
    """
    priority = 9

    def __init__(self,
                 warmup_min_iter: int = 1000,
                 start_factor: float = 0.,
                 warmup_epochs: int = 5,
                 min_lr_ratio: float = 0.0,
                 total_epochs: int = 360):

        self.warmup_min_iter = warmup_min_iter
        self.start_factor = start_factor
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs

        self._warmup_end = False
        self._base_lr = None

    def before_train(self, runner):
        """Operations before train.

        Args:
            runner (Runner): The runner of the training process.
        """
        optimizer = runner.optim_wrapper.optimizer
        for group in optimizer.param_groups:
            # If the param is never be scheduled, record the current value
            # as the initial value.
            group.setdefault('initial_lr', group['lr'])

        self._base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]
        self._min_lr = [i * self.min_lr_ratio for i in self._base_lr]

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch=None):
        """Operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
        """
        cur_iters = runner.iter
        optimizer = runner.optim_wrapper.optimizer
        dataloader_len = len(runner.train_dataloader)

        # The minimum warmup is self.warmup_min_iter
        warmup_total_iters = max(
            round(self.warmup_epochs * dataloader_len), self.warmup_min_iter)

        if cur_iters <= warmup_total_iters:
            # warm up
            alpha = cur_iters / warmup_total_iters
            factor = self.start_factor * (1 - alpha) + alpha

            for group_idx, param in enumerate(optimizer.param_groups):
                param['lr'] = self._base_lr[group_idx] * factor
        else:
            for group_idx, param in enumerate(optimizer.param_groups):
                total_iters = self.total_epochs * dataloader_len
                lr = self._min_lr[group_idx] + (
                    self._base_lr[group_idx] -
                    self._min_lr[group_idx]) * 0.5 * (
                        math.cos((cur_iters - warmup_total_iters) * math.pi /
                                 (total_iters - warmup_total_iters)) + 1.0)
                param['lr'] = lr
