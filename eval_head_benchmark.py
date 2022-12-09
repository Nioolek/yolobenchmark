import copy
import os
import torch
from mmengine import ConfigDict
import numpy as np

from mmyolo.engine import YoloCustomRunner
from mmyolo.engine.runners.yolo_custom_runner import PPYOLOE_collate_class_plus1, PPYOLOE_collate_class_plus
from mmyolo.models import PPYOLOE
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

model_depth = 0.33
model_width = 0.50

backbone_cfg = dict(
    type='PPYOLOECSPResNet',
    deepen_factor=model_depth,
    widen_factor=model_width,
    block_cfg=dict(
        type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
    norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
    act_cfg=dict(type='SiLU', inplace=True),
    attention_cfg=dict(
        type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
    use_large_stem=True)

neck_cfg = dict(
    type='PPYOLOECSPPAFPN',
    in_channels=[256, 512, 1024],
    out_channels=[192, 384, 768],
    deepen_factor=model_depth,
    widen_factor=model_width,
    num_csplayer=1,
    num_blocks_per_layer=3,
    block_cfg=dict(
        type='PPYOLOEBasicBlock', shortcut=False, use_alpha=False),
    norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
    act_cfg=dict(type='SiLU', inplace=True),
    drop_block_cfg=None,
    use_spp=True)

config_benchmark = dict(
    type='PPYOLOE',
    data_preprocessor=dict(
        type='PPYOLOEDetDataPreprocessor',
        mean=None,
        std=None,
        # bgr_to_rgb=True
    ),
    backbone=backbone_cfg,
    neck=neck_cfg,
    bbox_head=dict(
        type='PPYOLOEHead',
        in_channels=[192, 384, 768],
        width_mult=model_width,
        num_classes=80,
        fpn_strides=[8, 16, 32],
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        use_varifocal_loss=True,
    ),
    test_cfg=dict(
        agnostic=False,  # 是否区分类别进行 nms，False 表示要区分
        multi_label=False,  # 是否考虑多标签， 单张图检测是为 False，test 时候为 True，可以提高 1 个点的 mAP
        min_bbox_size=0,
        conf_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        # nms_top_k=1000,    # 在代码中已经写死，没有作为变量传入
        # max_per_img=300    # 在代码中已经写死，没有作为变量传入
    )
)

config_mmyolo = model = dict(
    type='PPYOLOE',
    data_preprocessor=dict(
        type='PPYOLOEDetDataPreprocessormmyolo',
        mean=None,
        std=None,
        # bgr_to_rgb=True
    ),
    backbone=backbone_cfg,
    neck=neck_cfg,
    bbox_head=dict(
        type='PPYOLOEHeadmmyolo',
        head_module=dict(
            type='PPYOLOEHeadModule',
            num_classes=80,
            in_channels=[192, 384, 768],
            widen_factor=model_width,
            featmap_strides=[8, 16, 32],
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
            act_cfg=dict(type='SiLU', inplace=True),
            num_base_priors=1),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False),
        # Since the average is implemented differently in the official
        # and mmdet, we're going to divide loss_weight by 4.
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.5 / 4)),
    train_cfg=dict(
        initial_epoch=30,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=80,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=80,
            topk=13,
            alpha=1,
            beta=6,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=1000,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300)
)

def collect_res(coll, batch):
    for pipe in coll.pipeline_list:
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

    data_mmyolo = {'inputs': torch.stack(imgs, 0), 'data_sample': torch.cat(labels_list, 0)}

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
    data_yolobenchmark = {'inputs': torch.stack(imgs, 0), 'data_sample': torch.stack(labels_list, 0)}
    return data_yolobenchmark, data_mmyolo

register_all_modules()

# 加载验证数据
data = torch.load('batch_data.pth')
model_yolobenchmark = MODELS.build(ConfigDict(config_benchmark))
# model_mmyolo = MODELS.build(ConfigDict(config_mmyolo))   # type: PPYOLOE
model_yolobenchmark.eval()
# model_mmyolo.eval()

# model_mmyolo.cuda()
model_yolobenchmark.cuda()


# # # 加载权重
# pth_path = 'ppyoloe_plus_crn_s_80e_coco.pth'
# state_dict = torch.load(pth_path)
# model_mmyolo.load_state_dict(state_dict['state_dict'], strict=False)

pth_path1 = 'ppyoloe_plus_crn_s_80e_coco_modellr.pth'
state_dict = torch.load(pth_path1)
model_yolobenchmark.load_state_dict(state_dict['state_dict'], strict=False)

data_yolobenchmark, data_mmyolo = collect_res(PPYOLOE_collate_class_plus(), data)

# # mmyolo推理搞好了
# data_mmyolo = model_mmyolo.data_preprocessor(data_mmyolo, True)
# x_mmyolo = model_mmyolo.extract_feat(data_mmyolo['inputs'])
# loss_mmyolo = model_mmyolo.bbox_head.loss(x_mmyolo, data_mmyolo['data_samples'])


data_yolobenchmark = model_yolobenchmark.data_preprocessor(data_yolobenchmark, True)
x_yolobenchmark = model_yolobenchmark.extract_feat(data_yolobenchmark[0])
loss_yolobenchmark = model_yolobenchmark.bbox_head.loss(x_yolobenchmark, data_yolobenchmark[1])

pass