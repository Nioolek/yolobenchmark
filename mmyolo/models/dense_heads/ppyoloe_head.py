import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import torchvision
from mmdet.utils import reduce_mean
from mmengine.structures import InstanceData
from torch import Tensor
from torch.cuda.amp import autocast

from mmyolo.registry import MODELS
from mmdet.structures import SampleList
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

# from ..assigner import ATSSAssigner, TaskAlignedAssigner
from ..assigner import ATSSAssigner, TaskAlignedAssigner
from ..layers.ppyoloe_brick import ConvBNLayer, get_activation


# from ppyoloe.bbox.utils import batch_distance2bbox, batch_distance2bbox_cxcywh
# from ..assigner import ATSSAssigner, TaskAlignedAssigner
# from .network_blocks import ConvBNLayer, get_activation
# from .losses import VarifocalLoss, FocalLoss, BboxLoss

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

def batch_distance2bbox(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox,
                               torch.zeros_like(out_bbox))
    return out_bbox


def batch_distance2bbox_cxcywh(points, distance, max_shapes=None):
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    cxcy = (x1y1+x2y2)/2
    wh = x2y2-x1y1
    out_bbox = torch.cat([cxcy, wh], -1)

    return out_bbox

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # 输入是cxcywh，转换成x1y1x2y2
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # prediction中排序是bbox,score,cls_conf
    multi_label = True
    output = [torch.zeros((0, 7), device=prediction.device) for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        if multi_label:
            ii, j = ((image_pred[:, 4:5] * image_pred[:, 5:5+num_classes]) > conf_thre).nonzero(as_tuple=False).T
            # mlvl_pred_map = torch.cat(
            #     (mlvl_pred_map[:, :4][i], mlvl_pred_map[i, j + 5, None],
            #      j[:, None].float()), 1)
            detections = torch.cat((image_pred[:, :5][ii], image_pred[ii, j+5, None], j[:, None].float()), 1)
        else:
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
        if not detections.size(0):
            continue
        elif detections.shape[0] > 1000:
            detections = detections[(detections[:, 4] * detections[:, 5]).argsort(descending=True)[:1000]]

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
        if nms_out_index.shape[0] > 300:
            nms_out_index = nms_out_index[:300]

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        super(GIoULoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clamp(0)
        h_inter = (ykis2 - ykis1).clamp(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def forward(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = torch.split(pbox, 1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, 1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


class VarifocalLoss(nn.Module):

    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self,
                pred_score,
                gt_score,
                label,
                alpha=0.75,
                gamma=2.0):
        """
        仅适用于当前任务。调用binary_cross_entropy不进行reduction。后乘上weight，再进行sum
        :param pred_score:
        :param gt_score:
        :param label:
        :param alpha:
        :param gamma:
        :return:
        """
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        with autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight.float()).sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = GIoULoss()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([1])
            loss_iou = torch.zeros([1])
            loss_dfl = torch.zeros([1])

        return loss_l1, loss_iou, loss_dfl

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


@MODELS.register_module()
class PPYOLOEHead(nn.Module):
    """
    由于回归方式不同，这里没法继承AnchorFreeHead。
    """
    def __init__(self,
                 in_channels=[1024, 512, 256],
                 width_mult=1.0,
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 use_varifocal_loss=True,
                 atss_topk=9,
                 train_cfg=None,
                 test_cfg=None):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.loss_weight = {
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5,
        }
        self.use_varifocal_loss = use_varifocal_loss
        self.varifocal_loss = VarifocalLoss().cuda()
        self.focal_loss = FocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max).cuda()
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        act = get_activation(act) if act is None or isinstance(act,
                                                               (str, dict)) else act

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

        self.use_tal = False
        self.atss_topk = atss_topk
        self.atss_assign = ATSSAssigner(9, num_classes=80)
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)

    def _init_weights(self, prior_prob=0.01):
        for conv in self.pred_cls:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.pred_reg:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = torch.nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)


    def forward_eval(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats, device=feats[0].device)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l]).permute(
                0, 2, 1, 3)
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, axis=-1)
        reg_dist_list = torch.cat(reg_dist_list, axis=-1)
        # torch.Size([1, 80, 8400]) torch.Size([1, 4, 8400]) torch.Size([8400, 2]) torch.Size([8400, 1])
        # print(cls_score_list.shape, reg_dist_list.shape, anchor_points.shape, stride_tensor.shape)

        # decode_outputs
        # [1,4,8400] x1y1x2y2
        # 为配合yolox输出是cxcywh，这里将输出转化
        # print(anchor_points.device, reg_dist_list.device)
        pred_bboxes = batch_distance2bbox_cxcywh(anchor_points,
                                          reg_dist_list.permute(0, 2, 1))
        pred_bboxes *= stride_tensor
        return torch.cat(  # 目标是(1,8400,85)
            [
                pred_bboxes,    #
                torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                cls_score_list.permute(0, 2, 1)
            ],
            axis=-1)

    def forward_train(self, feats):
        # anchors, anchor_points, num_anchors_list, stride_tensor = \
        #     self.generate_anchors_for_grid_cell(
        #         feats, self.fpn_strides, self.grid_cell_scale,
        #         self.grid_cell_offset, device=feats[0].device)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, axis=1)
        reg_distri_list = torch.cat(reg_distri_list, axis=1)

        return cls_score_list, reg_distri_list

    def _generate_anchors(self, feats=None, device='cuda:0'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_input_size[0] / stride)
                w = int(self.eval_input_size[1] / stride)
            shift_x = torch.arange(end=w, device=device) + self.grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            # anchor_point = paddle.cast(
            #     paddle.stack(
            #         [shift_x, shift_y], axis=-1), dtype='float32')
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward(self, feats):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats)
        else:
            return self.forward_eval(feats)

    def bbox_decode(self, anchor_points, pred_dist):
        batch_size, n_anchors, _ = pred_dist.shape
        pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj)
        return batch_distance2bbox(anchor_points, pred_dist)

    def loss(self, pred_maps, data_samples):
        cls_score_list, reg_distri_list = self(pred_maps)
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            self.generate_anchors_for_grid_cell(
                pred_maps, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset, device=pred_maps[0].device)
        losses = self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], data_samples)
        return losses

        # outs = self(pred_maps)
        # loss = self.get_loss(outs, data_samples)
        # return loss
    def get_loss(self, head_outs, targets):
        pred_scores, pred_distri, anchors, \
        anchor_points, num_anchors_list, stride_tensor = head_outs
        # 要确定一下targets的输入方式    xyxy

        # pred_bboxes:[batch_size, n_anchors, 4]
        # pred_scores:[batch_size, n_anchors, num_classes]
        # anchors:[n_anchors, 4]
        # anchor_points:[n_anchors, 2]
        # num_anchors_list:list [169,676,2704]先是检测大物体的特征图
        # stride_tensor:[n_anchors, 1]

        anchor_points_s = anchor_points / stride_tensor
        # x1y1x2y2
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)
        gt_labels = targets[:, :, :1]
        # xywh
        gt_bboxes = targets[:, :, 1:]
        pad_gt_mask = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        if not self.use_tal:
            assigned_labels, assigned_bboxes, assigned_scores = self.atss_assign(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1

        assigned_scores_sum = assigned_scores.sum()
        assigned_scores_sum = torch.clamp(reduce_mean(assigned_scores_sum), min=1)

        # if assigned_scores_sum < 5.:
        #     print('assigned_scores_sum < 5., we make the loss_cls to 0.')
        #     loss_cls = pred_scores.sum()*0
        #     loss_dfl = pred_distri.sum()*0
        #     loss_iou = pred_distri.sum()*0
        # else:

        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            loss_cls = self.varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self.focal_loss(pred_scores, assigned_scores, alpha_l)

        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                                                     assigned_labels, assigned_bboxes, assigned_scores,
                                                     assigned_scores_sum)

        # loss = self.loss_weight['class'] * loss_cls + \
        #        self.loss_weight['iou'] * loss_iou + \
        #        self.loss_weight['dfl'] * loss_dfl

        loss_dfl = self.loss_weight['dfl'] * loss_dfl
        loss_iou = self.loss_weight['iou'] * loss_iou
        loss_cls = self.loss_weight['class'] * loss_cls

        out_dict = {
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
        }
        return out_dict


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self, pred_maps, batch_img_metas, rescale=False):
        results = postprocess(pred_maps,
                              self.num_classes,
                              self.test_cfg.conf_thr,
                              self.test_cfg.nms.iou_threshold,
                              self.test_cfg.agnostic
                              )

        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(batch_img_metas)):
            result = results[img_id]
            if result.shape[0] == 0:
                pred_result = InstanceData()
                pred_result.bboxes = result[:, :4]
                pred_result.scores = result[:, 4] * result[:, 5]
                pred_result.labels = result[:, 6]
                result_list.append(pred_result)
                continue

            scale_factor = batch_img_metas[img_id]['scale_factor']
            if 'pad_param' in batch_img_metas[img_id]:
                pad_param = batch_img_metas[img_id]['pad_param']
            else:
                pad_param = None
            ori_shape = batch_img_metas[img_id]['ori_shape']

            det_bboxes = result[:, :4]
            if pad_param is not None:
                det_bboxes -= det_bboxes.new_tensor(
                    [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

            pred_result = InstanceData()
            pred_result.bboxes = det_bboxes[:, :4]

            clip_coords(pred_result.bboxes, ori_shape)

            pred_result.scores = result[:, 4] * result[:, 5]
            pred_result.labels = result[:, 6].int()

            result_list.append(pred_result)

        return result_list

    def generate_anchors_for_grid_cell(self, feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,
                                       device='cpu'):
        r"""
        Like ATSS, generate anchors based on grid size.
        Args:
            feats (List[Tensor]): shape[s, (b, c, h, w)]
            fpn_strides (tuple|list): shape[s], stride for each scale feature
            grid_cell_size (float): anchor size
            grid_cell_offset (float): The range is between 0 and 1.
        Returns:
            anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
            anchor_points (Tensor): shape[l, 2], "x, y" format.
            num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
            stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
        """
        assert len(feats) == len(fpn_strides)
        anchors = []
        anchor_points = []
        num_anchors_list = []
        stride_tensor = []
        for feat, stride in zip(feats, fpn_strides):
            _, _, h, w = feat.shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feat.dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feat.dtype)

            anchors.append(anchor.reshape([-1, 4]))
            anchor_points.append(anchor_point.reshape([-1, 2]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feat.dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).cuda()
        stride_tensor = torch.cat(stride_tensor).cuda()
        return anchors, anchor_points, num_anchors_list, stride_tensor


