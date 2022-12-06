# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmyolo.registry import HOOKS
from mmengine.runner import Runner
from mmengine.dataset import Compose
import copy

@HOOKS.register_module()
class YOLOXNewModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Defaults to 15.
       skip_type_keys (Sequence[str], optional): Sequence of type string to be
            skip pipeline. Defaults to ('Mosaic', 'RandomAffine', 'MixUp').
    """

    def __init__(
            self,
            num_last_epochs: int = 15,
            skip_type_keys: Sequence[str] = ('Mosaic', 'RandomAffine', 'MixUp')
    ) -> None:
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys

    def before_train_epoch(self, runner) -> None:
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:

            train_dataloader_cfg = copy.deepcopy(runner.cfg.train_dataloader)
            train_pipeline_cfg = train_dataloader_cfg.dataset.pipeline

            new_train_pipeline_cfg = []
            for transform in train_pipeline_cfg:
                if transform['type'] in self.skip_type_keys:
                    continue
                else:
                    new_train_pipeline_cfg.append(transform)

            # dataset does not need to be recreated
            runner.logger.info(f' New Pipeline: {new_train_pipeline_cfg}')
            train_loader.dataset.pipeline = Compose(new_train_pipeline_cfg)
            train_dataloader_cfg.dataset = train_loader.dataset

            new_train_dataloader = Runner.build_dataloader(train_dataloader_cfg)
            runner.train_loop.dataloader = new_train_dataloader

            runner.logger.info('recreate the dataloader!')
            runner.logger.info('Add additional L1 loss now!')
            model.bbox_head.use_l1 = True
