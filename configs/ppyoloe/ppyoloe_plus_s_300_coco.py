_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

runner_type = 'YoloCustomRunner'

img_scale = (640, 640)  # height, width

# dataset settings
data_root = '../data/coco/'
dataset_type = 'PPYOLOECocoDataset'

use_ceph = False
# load_from = 'CSPResNetb_s_pretrained1.pth'
# load_from = '/mnt/lustre/share_data/huanghaian/CSPResNetb_s_pretrained1.pth'
detect_mode = False
train_batch_size_pre_gpu = 8
train_num_workers = 8
val_batch_size_pre_gpu = 1
val_num_workers = 2
max_epoch = 80
save_epoch_interval = 10

model_depth = 0.33
model_width = 0.50

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
        multi_label=False,  # 是否考虑多标签， 单张图检测是为 False，test 时候为 True，可以提高 1 个点的 mAP
        min_bbox_size=0,
        conf_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        # nms_top_k=1000,    # 在代码中已经写死，没有作为变量传入
        # max_per_img=300    # 在代码中已经写死，没有作为变量传入
    )

model = dict(
    type='PPYOLOE',
    data_preprocessor=dict(
        type='PPYOLOEDetDataPreprocessor',
        mean=None,
        std=None,
        # bgr_to_rgb=True
    ),
    backbone=dict(
        type='PPYOLOEBackbone',
        return_idx=[1, 2, 3],
        use_large_stem=True,
        width_mult=model_width,
        depth_mult=model_depth
    ),
    neck=dict(
        type='PPYOLOECustomCSPPAN',
        out_channels=[768, 384, 192],
        stage_num=1,
        block_num=3,
        act='swish',
        spp=True,
        width_mult=model_width,
        depth_mult=model_depth
    ),
    bbox_head=dict(
        type='PPYOLOEHead',
        in_channels=[768, 384, 192],
        width_mult=model_width,
        num_classes=80,
        fpn_strides=[32, 16, 8],
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        use_varifocal_loss=True,
    ),
    test_cfg=model_test_cfg

)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    # # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PPYOLOEResize', target_size=img_scale, keep_ratio=False, interp=2),
    # ppyoloe中resize方式不维持ratio，所以单独用一个类做预处理，并且在这里转了RGB
    dict(type='PPYOLOENormalizeImage', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_scale=True),
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
    collate_fn=dict(type='PPYOLOE_collate_class'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        serialize_data=False,  # TODO
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        file_client_args=file_client_args
    ))

val_dataloader = dict(
    batch_size=val_batch_size_pre_gpu,
    num_workers=val_num_workers,
    persistent_workers=True if val_num_workers != 0 else False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        batch_size=val_batch_size_pre_gpu,
        file_client_args=file_client_args,
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        serialize_data=False,  # TODO
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file='annotations/instances_val2017.json',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.04, momentum=0.9, weight_decay=5e-4,
        nesterov=False),
    paramwise_cfg=dict(norm_decay_mult=0.,))

default_hooks = dict(
    param_scheduler=dict(type='PPYOLOELrUpdaterHook', total_epochs=360),
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
        momentum=0.0002,
        priority=49),
    dict(
        type='PPYOLOEAssignerHook', start_tal_epoch=100
    )
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

env_cfg = dict(cudnn_benchmark=False)