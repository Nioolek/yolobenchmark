import copy

from mmengine import ConfigDict

from mmengine.runner import Runner

from mmyolo.engine.runners.yolo_custom_runner import PPYOLOE_collate_class_plus, YoloCustomRunner
from mmyolo.utils import register_all_modules

train_batch_size_pre_gpu = 8
train_num_workers = 0
data_root = '../data/coco/'
dataset_type = 'PPYOLOECocoDataset'
file_client_args = dict(backend='disk')

train_dataloader = ConfigDict(dict(
    batch_size=train_batch_size_pre_gpu,
    num_workers=train_num_workers,
    persistent_workers=True if train_num_workers != 0 else False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='PPYOLOE_collate_class_plus'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        serialize_data=False,  # TODO
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        file_client_args=file_client_args
    )))

register_all_modules()

dataloader = YoloCustomRunner.build_dataloader(train_dataloader)
collate_fn = PPYOLOE_collate_class_plus()
for i in dataloader:
    raise NotImplementedError
