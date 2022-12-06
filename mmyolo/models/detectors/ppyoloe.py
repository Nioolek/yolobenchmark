# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmyolo.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmengine.dist import get_world_size
from mmengine.logging import print_log


@MODELS.register_module()
class PPYOLOE(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,
                 sync_bn=True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # SyncBatchNorm
        if sync_bn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')
