# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import cv2
import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData

from .box_level_augs.color_augs import color_aug_func
from .box_level_augs.geometric_augs import geometric_aug_func
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AIR_Resize(BaseTransform):
    def __init__(self, min_size_range, max_size):
        self.min_size = list(range(
            min_size_range[0],
            min_size_range[1] + 1
        ))
        self.max_size = max_size
        self.clip_object_border = True

    def get_size_ratio(self, image_size):
        target_size = random.choice(self.min_size)
        w, h = image_size
        t_w, t_h = target_size, target_size
        r = min(t_w / w, t_h / h)
        o_w, o_h = int(w * r), int(h * r)
        return (o_w, o_h)

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes.astype(np.float32)

    def transform(self, results):
        image = results['img']
        h, w = image.shape[:2]
        size = self.get_size_ratio((w, h))

        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        results['img'] = image
        results['img_shape'] = image.shape[:2]
        scale_factor = np.array([size[1] / h, size[0] / w], dtype=np.float32)
        results['scale_factor'] = scale_factor
        self._resize_bboxes(results)
        return results


@TRANSFORMS.register_module()
class AIR_SA_AUG(BaseTransform):
    def __init__(self, use_box_level=True):
        self.use_box_level = use_box_level
        autoaug_list = (6, 9, 5, 3,
                        3, 4, 2, 4,
                        4, 4, 5, 2,
                        4, 1, 4, 2,
                        6, 4, 2, 2,
                        2, 6, 2, 2,
                        2, 0, 5, 1,
                        3, 0, 8, 5,
                        2, 8, 7, 5,
                        1, 3, 3, 3)
        num_policies = 5
        self.scale_splits = [2048, 10240, 51200]
        self.box_prob = 0.3
        self.use_box_level = True
        self.use_box_color = True
        self.use_box_geo = True
        self.dynamic_scale_split = True

        box_aug_list = autoaug_list[4:]
        color_aug_types = list(color_aug_func.keys())
        geometric_aug_types = list(geometric_aug_func.keys())
        policies = []
        for i in range(num_policies):
            _start_pos = i * 6
            sub_policy = [(color_aug_types[box_aug_list[_start_pos + 0] % len(color_aug_types)],
                           box_aug_list[_start_pos + 1] * 0.1, box_aug_list[_start_pos + 2],),  # box_color policy
                          (geometric_aug_types[box_aug_list[_start_pos + 3] % len(geometric_aug_types)],
                           box_aug_list[_start_pos + 4] * 0.1, box_aug_list[_start_pos + 5])]  # box_geometric policy
            policies.append(sub_policy)

        _start_pos = num_policies * 6
        scale_ratios = {
            'area': [box_aug_list[_start_pos + 0], box_aug_list[_start_pos + 1], box_aug_list[_start_pos + 2]],
            'prob': [box_aug_list[_start_pos + 3], box_aug_list[_start_pos + 4], box_aug_list[_start_pos + 5]]}

        self.box_augs_dict = {'policies': policies, 'scale_ratios': scale_ratios}
        self.iteration = 0
        self.batch_size = 1
        self.num_workers = 1
        self.sa_init_flag = True

    def transform(self, results):
        iteration = self.iteration // self.batch_size * self.num_workers
        img = results['img']
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(img).float()
        boxes = results['gt_bboxes']
        labels = results['gt_bboxes_labels']
        flags = results['gt_ignore_flags']
        boxes = boxes[flags == False]
        labels = labels[flags == False]

        if self.use_box_level:
            target = InstanceData(metainfo={})
            target.bboxes = torch.from_numpy(boxes).float()
            target.labels = torch.from_numpy(labels).long()
            tensor, target = self.box_augs(tensor, target, iteration=iteration)
        results['img'] = tensor.numpy().transpose((1, 2, 0))
        results['gt_bboxes'] = target.bboxes.numpy()
        results['gt_bboxes_labels'] = target.labels.numpy()
        results['gt_ignore_flags'] = np.array([False for i in range(results['gt_bboxes'].shape[0])])

        self.iteration += 1
        return results
