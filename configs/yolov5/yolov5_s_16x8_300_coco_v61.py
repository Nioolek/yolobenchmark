_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

runner_type = 'YoloCustomRunner'

img_scale = (640, 640)  # height, width

# dataset settings
data_root = 'data/coco/'
dataset_type = 'YOLOV5CocoDataset'

use_ceph = True
detect_mode = False
train_batch_size_pre_gpu = 16
train_num_workers = 8
val_batch_size_pre_gpu = 1
val_num_workers = 2
max_epoch = 300
save_epoch_interval = 10

if use_ceph:
    file_client_args = dict(
        backend='petrel',
        path_mapping=dict({
            './data/': 's3://openmmlab/datasets/detection/',
            'data/': 's3://openmmlab/datasets/detection/'
        }))
else:
    file_client_args = dict(backend='disk')

if detect_mode:
    model_test_cfg = dict(
        agnostic=False,
        multi_label=False,
        min_bbox_size=0,
        conf_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=300)
else:
    model_test_cfg = dict(
        agnostic=False,  # 是否区分类别进行 nms，False 表示要区分
        multi_label=True,  # 是否考虑多标签， 单张图检测是为 False，test 时候为 True，可以提高 1 个点的 mAP
        min_bbox_size=0,
        conf_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300)

# model settings
model = dict(
    type='YOLOV5',
    data_preprocessor=dict(
        type='Yolov5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOV5Backbone', depth_multiple=0.33, width_multiple=0.5),
    bbox_head=dict(
        type='YOLOV5Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[0.33, 0.5, 1],
        anchor_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        featmap_strides=[32, 16, 8]),
    test_cfg=model_test_cfg)

if not detect_mode:
    test_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=file_client_args),
        # dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Yolov5Resize', scale=img_scale),
        dict(type='LetterResize', scale=img_scale, scaleup=False, auto=False),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor', 'pad_param'))
    ]
else:
    test_pipeline = [
        dict(type='LoadImageFromFile', file_client_args=file_client_args),
        # dict(type='LoadAnnotations', with_bbox=True),
        dict(type='LetterResize', scale=img_scale, scaleup=True, auto=True),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor', 'pad_param'))
    ]

train_dataloader = dict(
    batch_size=train_batch_size_pre_gpu,
    num_workers=train_num_workers,
    persistent_workers=True if train_num_workers != 0 else False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        serialize_data=False,  # TODO
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        file_client_args=file_client_args))

val_dataloader = dict(
    batch_size=val_batch_size_pre_gpu,
    num_workers=val_num_workers,
    persistent_workers=True if val_num_workers != 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        pad=0.5,
        data_root=data_root,
        batch_size=val_batch_size_pre_gpu,
        file_client_args=file_client_args,
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        serialize_data=False,  # TODO
        ann_file='annotations/instances_val2017.json',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005, nesterov=True,
                   batch_size=train_batch_size_pre_gpu),
    constructor='YOLOV5OptimizerConstructor'
)

default_hooks = dict(
    param_scheduler=dict(type='YOLOV5LrUpdaterHook', max_epoch=max_epoch),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_interval, max_keep_ckpts=2))

train_cfg = dict(max_epochs=max_epoch, val_interval=save_epoch_interval)

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='mmdet.ExpMomentumEMA',
#         momentum=0.0001,
#         update_buffers=True,
#         priority=49)
# ]
custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

env_cfg = dict(cudnn_benchmark=True)
