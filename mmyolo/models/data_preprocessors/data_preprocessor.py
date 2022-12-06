from mmdet.models.data_preprocessors import DetDataPreprocessor
from typing import Optional, Sequence, Tuple
import torch
from mmyolo.registry import MODELS


@MODELS.register_module()
class Yolov5DetDataPreprocessor(DetDataPreprocessor):
    def forward(self,
                data: dict,
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        if not training:
            return super().forward(data, training)

        inputs = data['inputs'].to(
            self.device, non_blocking=True).float() / 255
        data_samples = data['data_sample'].to(
            self.device, non_blocking=True)
        return inputs, data_samples


@MODELS.register_module()
class PPYOLOEDetDataPreprocessor(DetDataPreprocessor):
    def forward(self,
                data: dict,
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        if not training:
            return super().forward(data, training)

        inputs = data['inputs'].to(
            self.device, non_blocking=True).float()
        data_samples = data['data_sample'].to(
            self.device, non_blocking=True)
        return inputs, data_samples
