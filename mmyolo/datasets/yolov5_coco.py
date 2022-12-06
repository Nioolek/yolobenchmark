# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import random

import cv2
import mmcv
import mmengine
import numpy as np
import torch

from mmyolo.registry import DATASETS
from mmdet.datasets import CocoDataset


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


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation
    # https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1,
                                 (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def random_perspective(im,
                       targets=(),
                       segments=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0),
                       yolov7_style=False):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10),
    # translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective,
                             perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective,
                             perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations

    if yolov7_style:
        s = random.uniform(1 - scale, 1.1 + scale)
    else:
        s = random.uniform(1 - scale, 1 + scale)

    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate,
                             0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(
        0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] !=
                            0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(
                im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(
                im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = False
        new = np.zeros((n, 4))
        # False
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample # noqa
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :
                           2] / xy[:, 2:
                                      3] if perspective else xy[:, :
                                                                   2]  # perspective rescale or affine # noqa

                # clip
                new[i] = segment2box(xy, width, height)  # noqa

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] /
                  xy[:, 2:3] if perspective else xy[:, :2]).reshape(
                n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 1:5].T * s,
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def box_candidates(box1,
                   box2,
                   wh_thr=2,
                   ar_thr=20,
                   area_thr=0.1,
                   eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment,
    # box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 /
                                            (w1 * h1 + eps) > area_thr) & (
                   ar < ar_thr)  # candidates


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain
                                           ] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
                                            lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            # check_version(
            #     A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)
            ],
                bbox_params=A.BboxParams(
                    format='yolo',
                    label_fields=['class_labels']))

            # LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except Exception as e:
            print(e)
            # LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(
                image=im, bboxes=labels[:, 1:],
                class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array(
                [[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    # normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


@DATASETS.register_module()
class YOLOV5CocoDataset(CocoDataset):

    def __init__(self,
                 *args,
                 with_rectangular=True,
                 img_size=640,
                 batch_size=1,
                 stride=32,
                 pad=0.0,
                 file_client_args=None,
                 albu=True,
                 yolov7_style=False,
                 masaic9=True,
                 **kwargs):

        super(YOLOV5CocoDataset, self).__init__(*args, **kwargs)
        self.yolov7_style = yolov7_style
        self.img_size = img_size
        self.indices = range(len(self))
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.hyp = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        self.masaic9 = masaic9
        if self.yolov7_style:
            # yolov7l data / hyp.scratch.p5.yaml
            self.hyp = {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.2,
                'scale': 0.9,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.15,
                'copy_paste': 0.0
            }

        if albu:
            self.albumentations = Albumentations()
        else:
            self.albumentations = None

        self.file_client = None
        if file_client_args is not None:
            self.file_client = mmengine.FileClient(**file_client_args)

        # test 开启
        self.with_rectangular = with_rectangular
        if self.with_rectangular:

            image_shapes = self._calc_batch_shape()
            image_shapes = np.array(image_shapes, dtype=np.float64)

            n = len(image_shapes)  # number of images
            bi = np.floor(np.arange(n) / batch_size).astype(
                np.int)  # batch index
            nb = bi[-1] + 1  # number of batches
            self.batch = bi  # batch index of image

            ar = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
            irect = ar.argsort()

            self.data_list = [self.data_list[i] for i in irect]

            ar = ar[irect]
            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(
                np.array(shapes) * img_size / stride + pad).astype(
                np.int) * stride

    def _calc_batch_shape(self):
        batch_shape = []
        for data_info in self.data_list:
            batch_shape.append((data_info['width'], data_info['height']))
        return batch_shape

    def prepare_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        results = copy.deepcopy(self.data_list[idx])
        if self.with_rectangular:
            results['batch_shape'] = self.batch_shapes[self.batch[idx]]
        return self.pipeline(results)

    def _train(self, idx):
        # 要提前加载 label
        if self.yolov7_style and self.masaic9:
            if random.random() < 0.8:
                img, labels = self._load_mosaic(idx)
            else:
                img, labels = self._load_mosaic9(idx)
        else:
            img, labels = self._load_mosaic(idx)

        # MixUp augmentation
        if random.random() < self.hyp['mixup']:
            if self.yolov7_style and self.masaic9:
                if random.random() < 0.8:
                    img2, labels2 = self._load_mosaic(random.randint(0, len(self) - 1))
                else:
                    img2, labels2 = self._load_mosaic9(random.randint(0, len(self) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
            else:
                img, labels = mixup(
                    img, labels,
                    *self._load_mosaic(random.randint(0,
                                                      len(self) - 1)))

        nl = len(labels)  # number of labels
        if nl:
            if self.yolov7_style:
                clip = False
            else:
                clip = True

            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5],
                w=img.shape[1],
                h=img.shape[0],
                clip=clip,
                eps=1E-3)

        # Albumentations
        if self.albumentations:
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

        # HSV color-space
        augment_hsv(
            img,
            hgain=self.hyp['hsv_h'],
            sgain=self.hyp['hsv_s'],
            vgain=self.hyp['hsv_v'])

        # Flip left-right
        if random.random() < self.hyp['fliplr']:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        # Cutouts
        # labels = cutout(img, labels, p=0.5)

        # labels_out = torch.zeros((nl, 6))
        # if nl:
        #     labels_out[:, 1:] = torch.from_numpy(labels)

        return img, labels

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
            return None, (0, 0), (0, 0)

        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(
                img, (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def _load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images
        # into a 4-image mosaic, segments4 是 mask
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x))
                  for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(
            self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            for _ in range(50):
                img, (orig_h, orig_w), (h, w) = self._load_image(index)
                if img is None:
                    index = random.choices(self.indices, k=1)[0]
                else:
                    break

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # bug: Need to remove crowd bboxes
            # 读取 label
            gt_bboxes = []
            gt_bboxes_labels = []
            for instance in self.data_list[index]['instances']:
                if instance['ignore_flag'] == 0:
                    gt_bboxes.append(instance['bbox'])
                    gt_bboxes_labels.append(instance['bbox_label'])
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes_labels = np.array(gt_bboxes_labels)
            bboxes = gt_bboxes  # xyxy
            labels = gt_bboxes_labels

            # 对 bbox 进行处理
            if bboxes.shape[0] > 0:
                bboxes[:, 0::2] *= w / orig_w
                bboxes[:, 1::2] *= h / orig_h
                bboxes[:, 0::2] += padw
                bboxes[:, 1::2] += padh
            else:
                bboxes = np.empty((0, 4))

            labels = np.concatenate((labels[:, None], bboxes), axis=1)
            # Labels  segments
            # labels, segments = self.labels[index].copy(), self.segments[index].copy() # noqa
            # if labels.size:
            #     labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format # noqa
            # segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend([])

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # 需要 mask，故不使用
        # img4, labels4, segments4 = copy_paste(
        #     img4, labels4, segments4, p=self.hyp['copy_paste'])

        img4, labels4 = random_perspective(
            img4,
            labels4,
            segments4,
            degrees=self.hyp['degrees'],
            translate=self.hyp['translate'],
            scale=self.hyp['scale'],
            shear=self.hyp['shear'],
            perspective=self.hyp['perspective'],
            border=self.mosaic_border,
            yolov7_style=self.yolov7_style)  # border to remove

        return img4, labels4

    def _load_mosaic9(self, index):
        # loads images in a 9-mosaic

        labels9, segments9 = [], []
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
        for i, index in enumerate(indices):
            # Load image
            for _ in range(50):
                img, (orig_h, orig_w), (h, w) = self._load_image(index)
                if img is None:
                    index = random.choices(self.indices, k=1)[0]
                else:
                    break

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            # 读取 label
            gt_bboxes = []
            gt_bboxes_labels = []
            for instance in self.data_list[index]['instances']:
                if instance['ignore_flag'] == 0:
                    gt_bboxes.append(instance['bbox'])
                    gt_bboxes_labels.append(instance['bbox_label'])
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes_labels = np.array(gt_bboxes_labels)
            bboxes = gt_bboxes
            labels = gt_bboxes_labels

            # 对 bbox 进行处理
            if bboxes.shape[0] > 0:
                bboxes[:, 0::2] *= w / orig_w
                bboxes[:, 1::2] *= h / orig_h
                bboxes[:, 0::2] += padx
                bboxes[:, 1::2] += pady
            else:
                bboxes = np.empty((0, 4))

            labels = np.concatenate((labels[:, None], bboxes), axis=1)

            labels9.append(labels)
            segments9.extend([])

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        # 需要 mask，故不使用
        # img9, labels9, segments9 = copy_paste(
        #     img9, labels9, segments9, p=self.hyp['copy_paste'])
        img9, labels9 = random_perspective(img9, labels9, segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_img(idx)
        else:
            img, labels = self._train(idx)

            nl = len(labels)  # 归一化 class cx cy w h
            labels_out = torch.zeros((nl, 6))  # 归一化 batch_idx class cx cy w h
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels)

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            # print(labels_out.shape)

            return torch.from_numpy(img), labels_out

            # label 是归一化的 cxcywh 坐标，需要转化为 xyxy
            # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], img.shape[0],
            #                            img.shape[1])
            #
            # results = {
            #     'img': img,
            #     'gt_bboxes': labels[:, 1:].astype(np.float),
            #     'gt_bboxes_labels': labels[:, 0].astype(np.int),
            #     'img_shape': img.shape,
            #     'img_id': self.data_list[idx]['img_id'],
            #     'img_path': self.data_list[idx]['img_path'],
            #     'ori_height': 2,
            #     'ori_width': 2,
            #     'scale_factor': 1,
            #     'flip': 1,
            #     'flip_direction': 1,
            # }
            # return self.pipeline(results)
