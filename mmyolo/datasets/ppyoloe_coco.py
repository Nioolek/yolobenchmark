# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import random

import cv2
import mmcv
import numpy as np
import torch
from typing import List

from mmyolo.datasets.transforms.ppyoloe_operators import Decode, RandomDistort, RandomExpand, RandomCrop, RandomFlip
from mmyolo.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmengine.dataset.base_dataset import Compose
import mmengine

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def bbox_ioa(box1, box2, eps=1E-7):
    """Returns the intersection over box2 area given box1, box2.

    Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area



@DATASETS.register_module()
class PPYOLOECocoDataset(CocoDataset):

    def __init__(self,
                 *args,
                 img_size=640,
                 batch_size=1,
                 stide=32,
                 file_client_args=None,
                 **kwargs
                 ):
        super(CocoDataset, self).__init__(*args, **kwargs)
        self.pipeline_train = Compose([
            Decode(),
            RandomDistort(),
            RandomExpand(fill_value=[123.675, 116.28, 103.53]),
            RandomCrop(),
            RandomFlip()
        ])

        self.file_client = None
        if file_client_args is not None:
            self.file_client = mmengine.FileClient(**file_client_args)

    def prepare_img(self, idx):
        results = copy.deepcopy(self.data_list[idx])
        return self.pipeline(results)

    def _load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        path = self.data_list[index]['img_path']

        if self.file_client:
            img_bytes = self.file_client.get(path)
            img = mmcv.imfrombytes(img_bytes)
        else:
            img = cv2.imread(path)  # BGR

        if img is None:
            print('Image Not Found ' + path)
            return None, (0, 0)

        h0, w0 = img.shape[:2]  # orig hw
        # r = self.img_size / max(h0, w0)  # ratio
        # if r != 1:  # if sizes are not equal
        #     img = cv2.resize(
        #         img, (int(w0 * r), int(h0 * r)),
        #         interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0)     #, img.shape[:2]

    def _train(self, idx):
        img, (orig_h, orig_w) = self._load_image(idx)   # BGR
        gt_bboxes = []   # x1y1x2y2
        gt_bboxes_labels = []
        for instance in self.data_list[idx]['instances']:
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_bboxes_labels.append(instance['bbox_label'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_bboxes_labels = np.array(gt_bboxes_labels)
        bboxes = gt_bboxes  # xyxy
        labels = gt_bboxes_labels
        # labels = np.concatenate((labels[:, None], bboxes), axis=1)

        results = {
            'img': img,
            'gt_bbox': bboxes,
            'gt_class': labels
        }
        results = self.pipeline_train(results)
        # labels = np.concatenate((results['gt_class'][:, None], results['gt_bbox']), axis=1)

        # return results['img'], labels
        return results



    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_img(idx)
        else:
            results = self._train(idx)

            # img, labels = self._train(idx)
            #
            # nl = len(labels)  # 归一化 class cx cy w h
            # labels_out = torch.zeros((nl, 6))  # 归一化 batch_idx class cx cy w h
            # if nl:
            #     labels_out[:, 1:] = torch.from_numpy(labels)
            #
            # # Convert
            # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # img = np.ascontiguousarray(img)
            # # print(labels_out.shape)

            # return torch.from_numpy(img), labels_out
            return results


    def filter_data(self) -> List   [dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
            if filter_empty_gt:
                print('filter empty gt')
                min_size = self.filter_cfg.get('min_size', 0)

                # obtain images that contain annotation
                ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
                # obtain images that contain annotations of the required categories
                ids_in_cat = set()
                for i, class_id in enumerate(self.cat_ids):
                    ids_in_cat |= set(self.cat_img_map[class_id])
                # merge the image id sets of the two conditions and use the merged set
                # to filter out images if self.filter_empty_gt=True
                ids_in_cat &= ids_with_ann

                valid_data_infos = []
                for i, data_info in enumerate(self.data_list):
                    img_id = data_info['img_id']
                    width = data_info['width']
                    height = data_info['height']
                    if filter_empty_gt and img_id not in ids_in_cat:
                        continue
                    if min(width, height) >= min_size:
                        valid_data_infos.append(data_info)
            else:
                return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos

