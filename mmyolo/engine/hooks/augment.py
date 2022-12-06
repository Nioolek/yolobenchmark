# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

from mmengine.hooks import Hook

from mmyolo.datasets.transforms.box_level_augs.box_level_augs import Box_augs
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class SA_AUG_Hook(Hook):

    def __init__(self, iter_epochs):
        super(SA_AUG_Hook, self).__init__()
        self.iter_epochs=iter_epochs

    def before_run(self, runner):
        pipelines = runner.train_dataloader.dataset.pipeline
        for p in pipelines.transforms:
            if 'AIR_SA_AUG' in p.__str__():
                if p.sa_init_flag:
                    self.iters_per_epoch = len(runner.train_dataloader)
                    p.max_iters = self.iter_epochs * self.iters_per_epoch

                    p.box_augs = Box_augs(box_augs_dict=p.box_augs_dict, max_iters=p.max_iters,
                                             scale_splits=p.scale_splits,
                                             box_prob=p.box_prob, dynamic_scale_split=p.dynamic_scale_split,
                                             use_color=p.use_box_color, use_geo=p.use_box_geo)

                    p.iteration = runner.epoch * len(runner.train_dataloader)
                    p.batch_size = runner.train_dataloader.batch_size
                    p.num_workers = max(1, runner.train_dataloader.num_workers)
                    print("===box_augs has been replaced===")

                    p.sa_init_flag = False
