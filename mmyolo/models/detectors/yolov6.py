# Copyright (c) OpenMMLab. All rights reserved.
from mmyolo.registry import MODELS
from mmdet.models.detectors.single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOV6(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
