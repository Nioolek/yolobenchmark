_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

img_scale = (640, 640)  # height, width

# dataset settings
data_root = 'data/coco/'
dataset_type = 'YOLOXCocoDataset'

use_ceph = True
train_batch_size_pre_gpu = 8
train_num_workers = 8
val_batch_size_pre_gpu = 8
val_num_workers = 2
base_lr = 0.01
max_epochs = 300
num_last_epochs = 15
interval = 10
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

# model settings
model = dict(
    type='AIRDet',
    sync_bn=True,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='CSPDarknet',
        dep_mul=0.33,
        wid_mul=0.5
    ),
    neck=dict(
        type='GiraffeNeck',
        min_level=3,
        max_level=5,
        num_levels=3,
        act_type="silu",
        fpn_config=None,
        fpn_name="giraffeneck",
        fpn_channels=[96, 160, 384],
        out_fpn_channels=[96, 160, 384],
        weight_method="concat",
        depth_multiplier=2,
        width_multiplier=1.0,
        with_backslash=True,
        with_slash=True,
        with_skip_connect=True,
        skip_connect_type="log2n",
        separable_conv=False,
        feature_info=[dict(num_chs=128, reduction=8), dict(num_chs=256, reduction=16), dict(num_chs=512, reduction=32)],
        merge_type="csp",
        pad_type='',
        downsample_type="max",
        upsample_type="nearest",
        apply_resample_bn=True,
        conv_after_downsample=False,
        redundant_bias=False,
        conv_bn_relu_pattern=False,
        alternate_init=False
    ),
    bbox_head=dict(
        type='GFocalHead_Tiny',
        num_classes=80,
        in_channels=[96, 160, 384],
        reg_channels=64,
        feat_channels=96,
        reg_max=14,
        add_mean=True,
        norm="bn",
        act="silu",
        start_kernel_size=3,
        conv_groups=2,
        conv_type="BaseConv",
        octbase=5,
        l1_switch="False"
    ),
    train_cfg=dict(assigner=dict(type='AIRDETSimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.05, nms=dict(type='nms', iou_threshold=0.7), max_per_img=500))

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args, imdecode_backend='pillow'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(type='NewMosaic', img_scale=img_scale, pad_val=114.0, pre_pipeline=[
        dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args, imdecode_backend='pillow'),
        dict(type='mmdet.LoadAnnotations', with_bbox=True),
    ]),
    dict(
        type='mmdet.RandomAffine',
        max_rotate_degree=10.0,
        max_translate_ratio=0.1,
        max_shear_degree=2.0,
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='NewMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_pipeline=[
            dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args, imdecode_backend='pillow'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True),
        ]),
    dict(type='AIR_Resize', min_size_range=(448, 832), max_size=640),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='AIR_SA_AUG'),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    # pillow vs cv2
    dict(type='mmdet.LoadImageFromFile', file_client_args=file_client_args, imdecode_backend='pillow'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    # images_per_batch=1: 44.1; images_per_batch=8: 44.2; official=64
    batch_size=val_batch_size_pre_gpu,
    num_workers=val_num_workers,
    persistent_workers=True if val_num_workers != 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

# TODO： Verify if weight_decay is to be adaptively scaled
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    constructor='YOLOV5OptimizerConstructor')

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))
train_cfg = dict(max_epochs=max_epochs, val_interval=save_epoch_interval,
                 dynamic_intervals=[(max_epochs - num_last_epochs, 1)])

custom_hooks = [
    dict(
        type='SA_AUG_Hook',
        iter_epochs=max_epochs - num_last_epochs,
        priority=47),
    dict(
        type='YOLOXNewModeSwitchHook',
        num_last_epochs=num_last_epochs,
        skip_type_keys=('NewMosaic', 'mmdet.RandomAffine', 'NewMixUp', 'AIR_SA_AUG'),
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0002,
        priority=49)
]
