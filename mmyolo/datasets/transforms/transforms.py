# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmyolo.registry import TRANSFORMS
from typing import List, Optional, Sequence, Tuple, Union
from mmdet.datasets.transforms import MixUp, Mosaic
import collections
import copy
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.dataset.base_dataset import Compose
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from numpy import random


@TRANSFORMS.register_module()
class Yolov5Resize(BaseTransform):

    def __init__(self, scale=None, backend='cv2'):
        self.backend = backend
        self.img_scale = scale

    def transform(self, results):
        img = results['img']
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_scale[0] / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                img, (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)

        # 是否有必要使用下面两行, 在对齐推理精度时候，yolov5 可以关闭，yolov6 如果关闭会差距 0.1
        h, w = img.shape[:2]
        r = h / h0

        scale_factor = np.array([r, r, r, r], dtype=np.float32)
        results['scale_factor'] = scale_factor
        results['img'] = img
        results['img_shape'] = img.shape

        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            gt_bboxes *= scale_factor

        return results


# from https://github.com/ultralytics/yolov5

# 第一步： 计算缩放比例，假设input_shape = (181, 110, 3)，输出shape=201，先计算缩放比例1.11和1.9,选择小比例
#         这个是常规操作，保证缩放后最长边不超过设定值
# 第二步： 计算pad像素，前面resize后会变成(201,122,3)，理论上应该pad=(0,79)，采用最小pad原则，设置最多不能pad超过64像素
#         故对79采用取模操作，变成79%64=15，然后对15进行/2，然后左右pad即可
# 原因是：在单张推理时候不想用letterbox的正方形模式，而是矩形模式，可以加快推理时间、但是在batch测试中，会右下pad到整个batch内部wh最大值


@TRANSFORMS.register_module()
class LetterResize(BaseTransform):

    def __init__(self,
                 scale=None,
                 color=(114, 114, 114),
                 auto=True,
                 scaleFill=False,
                 scaleup=True,
                 backend='cv2'):
        self.image_size_hw = scale
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.backend = backend

    # 暂时没有对 bbox 进行处理
    def transform(self, results):
        img = results['img']

        if 'batch_shape' in results:
            batch_shape = results['batch_shape']
            self.image_size_hw = batch_shape

        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(self.image_size_hw, int):
            self.image_size_hw = (self.image_size_hw, self.image_size_hw)

        # Scale ratio (new / old)
        r = min(self.image_size_hw[0] / shape[0],
                self.image_size_hw[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)
        ratio = r, r
        # 保存图片宽高缩放的最佳size
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        # 为了得到指定输出size，可能需要pad,pad参数
        dw, dh = self.image_size_hw[1] - new_unpad[0], self.image_size_hw[
            0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            # 直接强制拉伸成指定输出
            new_unpad = (self.image_size_hw[1], self.image_size_hw[0])
            ratio = self.image_size_hw[1] / shape[1], self.image_size_hw[
                0] / shape[0]  # width, height ratios

        # 左右padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # 没有padding前
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        results['img_shape'] = img.shape
        scale_factor = np.array([ratio[0], ratio[1], ratio[0], ratio[1]],
                                dtype=np.float32)

        if 'scale_factor' in results:
            results['scale_factor'] *= scale_factor
        else:
            results['scale_factor'] = scale_factor

        # padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.color)  # add border

        # 在对齐推理精度时候，实测使用上下代码性能没有差异
        results['img'] = img
        results['pad_param'] = np.array([top, bottom, left, right],
                                        dtype=np.float32)

        # 保留小数，会影响性能
        # results['pad_param'] = np.array([dh,
        #                                  dh,
        #                                  dw,
        #                                  dw],
        #                                 dtype=np.float32)

        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            gt_bboxes *= scale_factor
            gt_bboxes[:, 0::2] += left
            gt_bboxes[:, 1::2] += top
        return results


@TRANSFORMS.register_module()
class NewMixUp(MixUp):

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 max_iters: int = 15,
                 bbox_clip_border: bool = True,
                 pre_pipeline: Sequence[str] = None,
                 max_refetch=15) -> None:

        super(NewMixUp, self).__init__(img_scale=img_scale,
                                       ratio_range=ratio_range,
                                       flip_ratio=flip_ratio,
                                       pad_val=pad_val,
                                       max_iters=max_iters,
                                       bbox_clip_border=bbox_clip_border)

        pipeline = []
        for transform in pre_pipeline:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                pipeline.append(transform)
        self.pre_pipeline = Compose(pipeline)
        self.max_refetch = max_refetch

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset))
            gt_bboxes_i = dataset.get_data_info(index)['instances']
            if len(gt_bboxes_i) != 0:
                break

        return index

    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'dataset' in results

        # Be careful: deep copying can be very time-consuming if results includes dataset.
        dataset = results.pop('dataset')

        for _ in range(self.max_refetch):
            indexes = self.get_indexes(dataset)
            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]

            mix_data_infos = [dataset.get_data_info(index) for index in indexes]
            mix_results = [
                copy.deepcopy(self.pre_pipeline(data)) for data in mix_data_infos
            ]
            if None not in mix_results:
                results['mix_results'] = mix_results
                break
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')

        for i in range(self.max_refetch):
            # To confirm the results passed the training pipeline
            # of the wrapper is not None.
            updated_results = super().transform(copy.deepcopy(results))
            if updated_results is not None:
                results = updated_results
                break
        else:
            raise RuntimeError(
                'The training pipeline of the dataset wrapper'
                ' always return None.Please check the correctness '
                'of the dataset and its pipeline.')

        if 'mix_results' in results:
            results.pop('mix_results')

        results['dataset'] = dataset

        return results


@TRANSFORMS.register_module()
class NewMosaic(Mosaic):

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 prob: float = 1.0,
                 pre_pipeline: Sequence[str] = None,
                 max_refetch=15) -> None:
        super().__init__(img_scale=img_scale,
                         center_ratio_range=center_ratio_range,
                         bbox_clip_border=bbox_clip_border,
                         pad_val=pad_val,
                         prob=prob)

        pipeline = []
        for transform in pre_pipeline:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                pipeline.append(transform)
        self.pre_pipeline = Compose(pipeline)
        self.max_refetch = max_refetch

    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'dataset' in results
        # Be careful: deep copying can be very time-consuming if results includes dataset.
        dataset = results.pop('dataset')

        for _ in range(self.max_refetch):
            indexes = super().get_indexes(dataset)
            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]
            mix_data_infos = [dataset.get_data_info(index) for index in indexes]
            mix_results = [
                copy.deepcopy(self.pre_pipeline(data)) for data in mix_data_infos
            ]
            if None not in mix_results:
                results['mix_results'] = mix_results
                break
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')

        for i in range(self.max_refetch):
            # To confirm the results passed the training pipeline
            # of the wrapper is not None.
            updated_results = super().transform(copy.deepcopy(results))
            if updated_results is not None:
                results = updated_results
                break
        else:
            raise RuntimeError(
                'The training pipeline of the dataset wrapper'
                ' always return None.Please check the correctness '
                'of the dataset and its pipeline.')

        if 'mix_results' in results:
            results.pop('mix_results')

        results['dataset'] = dataset
        return results
