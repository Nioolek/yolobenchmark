import cv2
import os
import numpy as np
from mmcv.transforms.base import BaseTransform

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PPYOLOEResize(BaseTransform):
    def __init__(self,
                 target_size=(640, 640),
                 keep_ratio=True,
                 interp=2):
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    # def apply_segm(self, segms, im_size, scale):
    #     def _resize_poly(poly, im_scale_x, im_scale_y):
    #         resized_poly = np.array(poly).astype('float32')
    #         resized_poly[0::2] *= im_scale_x
    #         resized_poly[1::2] *= im_scale_y
    #         return resized_poly.tolist()
    #
    #     def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
    #         if 'counts' in rle and type(rle['counts']) == list:
    #             rle = mask_util.frPyObjects(rle, im_h, im_w)
    #
    #         mask = mask_util.decode(rle)
    #         mask = cv2.resize(
    #             mask,
    #             None,
    #             None,
    #             fx=im_scale_x,
    #             fy=im_scale_y,
    #             interpolation=self.interp)
    #         rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    #         return rle
    #
    #     im_h, im_w = im_size
    #     im_scale_x, im_scale_y = scale
    #     resized_segms = []
    #     for segm in segms:
    #         if is_poly(segm):
    #             # Polygon format
    #             resized_segms.append([
    #                 _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
    #             ])
    #         else:
    #             # RLE format
    #             import pycocotools.mask as mask_util
    #             resized_segms.append(
    #                 _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))
    #
    #     return resized_segms

    def transform(self, results):
        img = results['img']
        h0, w0 = img.shape[:2]  # orig hw
        im_shape = img.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]
        im = self.apply_image(results['img'], [im_scale_x, im_scale_y])
        results['img'] = im

        results['img_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in results:
            scale_factor = results['scale_factor']
            results['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x,
                 scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            results['scale_factor'] = np.asarray(
                [im_scale_x, im_scale_y, im_scale_x, im_scale_y], dtype=np.float32)

        if 'gt_bboxes' in results:
            results['gt_bbox'] = self.apply_bbox(results['gt_bbox'],
                                                 [im_scale_x, im_scale_y],
                                                 [resize_w, resize_h])
        return results


@TRANSFORMS.register_module()
class PPYOLOENormalizeImage(BaseTransform):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale

    def transform(self, results):
        im = results['img']
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)    # RGB
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0

        im -= mean
        im /= std

        results['img'] = im
        results['pad_param'] = np.array([0, 0, 0, 0],
                                        dtype=np.float32)
        return results
