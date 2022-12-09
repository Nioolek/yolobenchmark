try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral
import os
import cv2
import torch
from mmengine.dataset import worker_init_fn
from mmengine.dist import get_rank
from mmengine.runner import Runner
from torch.utils.data import DataLoader
import copy
import uuid
import numpy as np
from functools import partial
from mmyolo.registry import DATA_SAMPLERS, DATASETS, RUNNERS

cv2.setNumThreads(
    0
)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
NUM_THREADS = min(8, max(1,
                         os.cpu_count() -
                         1))  # number of YOLOv5 multiprocessing threads
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['OMP_NUM_THREADS'] = str(
    NUM_THREADS)  # OpenMP max threads (PyTorch and SciPy)

print(
    f'===========os.cpu_count():{os.cpu_count()},NUM_THREADS: {NUM_THREADS}====================='
)


def yolov5_collate_fn(batch):
    im, label = zip(*batch)  # transposed
    for i, lb in enumerate(label):
        lb[:, 0] = i  # add target image index for build_targets()
    return {'inputs': torch.stack(im, 0), 'data_sample': torch.cat(label, 0)}
    # return torch.stack(im, 0), torch.cat(label, 0)


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
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

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['img']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))

        # apply image
        im_shape = im.shape
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

        im = self.apply_image(sample['img'], [im_scale_x, im_scale_y])
        sample['img'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])
        return sample


class BatchRandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True,
                 random_interp=False):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp

        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples, context=None):
        if self.random_size:
            index = np.random.choice(len(self.target_size))
            target_size = self.target_size[index]
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=interp)
        return resizer(samples, context=context)


class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True, norm_type='mean_std'):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type= norm_type
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['img']
        im = im.astype(np.float32, copy=False)
        # mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        # std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std

        sample['img'] = im
        return sample


class Permute(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['img']
        im = im.transpose((2, 0, 1))
        sample['img'] = im
        return sample

class PPYOLOE_collate_class():

    def __init__(self):
        self.pipeline_list = [
            BatchRandomResize(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size=True, random_interp=True, keep_ratio=False),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_scale=True),
            Permute()
        ]
        
    def __call__(self, batch):
        for pipe in self.pipeline_list:
            batch = pipe(batch)

        num_max_boxes = max([len(s['gt_bbox']) for s in batch])
        imgs = []
        labels_list = []
        for ind, i in enumerate(batch):
            img = i['img']
            img = np.ascontiguousarray(img)

            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.float32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            num_gt = len(i['gt_bbox'])
            if num_gt > 0:
                pad_gt_class[:num_gt] = i['gt_class'][:, None]
                pad_gt_bbox[:num_gt] = i['gt_bbox']
            labels = np.concatenate((pad_gt_class, pad_gt_bbox), axis=1)
            imgs.append(torch.from_numpy(img))
            labels_list.append(torch.from_numpy(labels))
        return {'inputs': torch.stack(imgs, 0), 'data_sample': torch.stack(labels_list, 0)}


class PPYOLOE_collate_class_plus():

    def __init__(self):
        self.pipeline_list = [
            BatchRandomResize(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
                              random_size=True, random_interp=True, keep_ratio=False),
            NormalizeImage(mean=[0., 0., 0.], std=[1., 1., 1.], is_scale=True, norm_type=None),
            Permute()
        ]

    def __call__(self, batch):
        # print('!!! save')
        # torch.save(batch, 'batch_data.pth')

        for pipe in self.pipeline_list:
            batch = pipe(batch)

        num_max_boxes = max([len(s['gt_bbox']) for s in batch])
        imgs = []
        labels_list = []
        for ind, i in enumerate(batch):
            img = i['img']
            img = np.ascontiguousarray(img)

            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.float32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            num_gt = len(i['gt_bbox'])
            if num_gt > 0:
                pad_gt_class[:num_gt] = i['gt_class'][:, None]
                pad_gt_bbox[:num_gt] = i['gt_bbox']
            labels = np.concatenate((pad_gt_class, pad_gt_bbox), axis=1)
            imgs.append(torch.from_numpy(img))
            labels_list.append(torch.from_numpy(labels))
        return {'inputs': torch.stack(imgs, 0), 'data_sample': torch.stack(labels_list, 0)}

class PPYOLOE_collate_class_plus1():

    def __init__(self):
        self.pipeline_list = [
            BatchRandomResize(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
                              random_size=True, random_interp=True, keep_ratio=False),
            NormalizeImage(mean=[0., 0., 0.], std=[1., 1., 1.], is_scale=True, norm_type=None),
            Permute()
        ]

    def __call__(self, batch):
        for pipe in self.pipeline_list:
            batch = pipe(batch)

        num_max_boxes = max([len(s['gt_bbox']) for s in batch])
        imgs = []
        labels_list = []
        for ind, i in enumerate(batch):
            img = i['img']
            img = np.ascontiguousarray(img)

            gt_class = torch.from_numpy(i['gt_class'][:, None]).float()
            gt_bbox = torch.from_numpy(i['gt_bbox'])
            batch_idx = gt_class.new_full((len(gt_class), 1), ind)
            bboxes_labels = torch.cat((batch_idx, gt_class, gt_bbox), dim=1)
            labels_list.append(bboxes_labels)
            imgs.append(torch.from_numpy(img))

            #
            # pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.float32)
            # pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            # num_gt = len(i['gt_bbox'])
            # if num_gt > 0:
            #     pad_gt_class[:num_gt] = i['gt_class'][:, None]
            #     pad_gt_bbox[:num_gt] = i['gt_bbox']
            # labels = np.concatenate((pad_gt_class, pad_gt_bbox), axis=1)
            # imgs.append(torch.from_numpy(img))
            # labels_list.append(torch.from_numpy(labels))
        return {'inputs': torch.stack(imgs, 0), 'data_sample': torch.cat(labels_list, 0)}

@RUNNERS.register_module()
class YoloCustomRunner(Runner):
    @staticmethod
    def build_dataloader(dataloader,
                         seed=None,
                         diff_rank_seed=False):
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # Don't change test process
        if dataloader_cfg.dataset.get('test_mode', False):
            return Runner.build_dataloader(dataloader, seed, diff_rank_seed)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        if seed is not None:
            init_fn = partial(
                worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=get_rank(),
                seed=seed)
        else:
            init_fn = None

        # change
        collate_fn_cfg = dataloader_cfg.pop('collate_fn', None)
        if collate_fn_cfg and collate_fn_cfg['type'] == 'PPYOLOE_collate_class':
            collate_fn = PPYOLOE_collate_class()
        elif collate_fn_cfg and collate_fn_cfg['type'] == 'PPYOLOE_collate_class_plus':
            collate_fn = PPYOLOE_collate_class_plus()
        elif collate_fn_cfg and collate_fn_cfg['type'] == 'PPYOLOE_collate_class_plus1':
            collate_fn = PPYOLOE_collate_class_plus1()
        else:
            collate_fn = yolov5_collate_fn

        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,  # change
            worker_init_fn=init_fn,
            pin_memory=True,  # change
            **dataloader_cfg)
        return data_loader
