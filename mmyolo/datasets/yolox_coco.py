# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import random

import cv2
import mmcv
import numpy as np
import torch

from mmyolo.registry import DATASETS
from mmdet.datasets import CocoDataset
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union


@DATASETS.register_module()
class YOLOXCocoDataset(CocoDataset):

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        return self.pipeline(data_info)