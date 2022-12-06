import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.airdet_brick import BaseConv, DWConv
from mmyolo.models.task_modules.assigners import AIRDETSimOTAAssigner
from mmengine.model import normal_init, bias_init_with_prob
from mmdet.structures.bbox import bbox_overlaps
# from mmdet.models.layers import multiclass_nms
from mmdet.utils import reduce_mean
from mmdet.models.utils import multi_apply
from mmdet.models.losses import GIoULoss, DistributionFocalLoss, QualityFocalLoss
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmyolo.registry import MODELS
import torchvision
from mmdet.models.utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
import functools


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=100,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores
    # filter out boxes with low scores
    valid_mask = scores > score_thr  # 1000 * 80 bool

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    # bboxes -> 1000, 4
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)  # mask->  1000*80*4, 80000*4
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        scores = multi_bboxes.new_zeros((0,))

        return bboxes, scores, labels

    keep = torchvision.ops.batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    return bboxes[keep], scores[keep], labels[keep]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        b, nb, ne, _ = x.size()
        x = x.reshape(b * nb * ne, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, nb, 4)
        return x


@MODELS.register_module()
class GFocalHead_Tiny(BaseDenseHead):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,  # 4
                 feat_channels=256,
                 reg_max=12,
                 reg_topk=4,
                 reg_channels=64,
                 strides=[8, 16, 32],
                 add_mean=True,
                 norm='gn',
                 act='relu',
                 start_kernel_size=3,
                 conv_groups=1,
                 conv_type='BaseConv',
                 simOTA_cls_weight=1.0,
                 simOTA_iou_weight=3.0,
                 octbase=8,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.strides = strides
        self.feat_channels = feat_channels if isinstance(feat_channels, list) \
            else [feat_channels] * len(self.strides)

        self.cls_out_channels = num_classes + 1  # add 1 for keep consistance with former models
        # and will be deprecated in future.
        self.stacked_convs = stacked_convs
        self.conv_groups = conv_groups
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.start_kernel_size = start_kernel_size

        self.norm = norm
        self.act = act
        self.conv_module = DWConv if conv_type == 'DWConv' else BaseConv

        if add_mean:
            self.total_dim += 1

        self.assigner = AIRDETSimOTAAssigner(
            center_radius=2.5,
            cls_weight=simOTA_cls_weight,
            iou_weight=simOTA_iou_weight,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))

        super(GFocalHead_Tiny, self).__init__()
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False, beta=2.0, loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        self.relu = nn.ReLU(inplace=True)
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else self.start_kernel_size
            cls_convs.append(
                self.conv_module(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=self.conv_groups,
                    norm=self.norm,
                    act=self.act))
            reg_convs.append(
                self.conv_module(
                    chn,
                    feat_channels,
                    kernel_size,
                    stride=1,
                    groups=self.conv_groups,
                    norm=self.norm,
                    act=self.act))

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
        reg_conf = nn.Sequential(*conf_vector)

        return cls_convs, reg_convs, reg_conf

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_confs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs, reg_conf = self._build_not_shared_convs(
                self.in_channels[i],
                self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.reg_confs.append(reg_conf)

        self.gfl_cls = nn.ModuleList(
            [nn.Conv2d(
                self.feat_channels[i],
                self.cls_out_channels,
                3,
                padding=1) for i in range(len(self.strides))])

        self.gfl_reg = nn.ModuleList(
            [nn.Conv2d(
                self.feat_channels[i],
                4 * (self.reg_max + 1),
                3,
                padding=1) for i in range(len(self.strides))])

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conf in self.reg_confs:
            for m in reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction

            - cls_scores (list[Tensor]): Classification and quality (IoU)
              joint scores for all scale levels, each is a 4D-tensor,
              the channel number is num_classes.
            - bbox_preds (list[Tensor]): Box distribution logits for all
              scale levels, each is a 4D-tensor, the channel number is
              4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single,
                           x,
                           self.cls_convs,
                           self.reg_convs,
                           self.gfl_cls,
                           self.gfl_reg,
                           self.reg_confs,
                           self.scales)

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, reg_conf, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = reg_conf(stat.reshape(N, 4 * self.total_dim, H, W))
        cls_score = gfl_cls(cls_feat).sigmoid() * quality_score

        return cls_score, bbox_pred

    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        batch_img_metas=None,
                        cfg=None,
                        rescale: bool = False,
                        with_nms: bool = True):

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        cfg = self.test_cfg if cfg is None else cfg

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)[..., :self.num_classes]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(
                num_imgs,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=flatten_cls_scores.device)
            for i, stride in enumerate(self.strides)]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        dis_preds = self.integral(flatten_bbox_preds) * mlvl_priors[..., 2, None]
        bboxes = distance2bbox(mlvl_priors[..., :2], dis_preds)

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_bboxes = bboxes[img_id]
            multi_scores = flatten_cls_scores[img_id]
            det_bboxes, scores, labels = multiclass_nms(multi_bboxes, multi_scores, cfg.score_thr, cfg.nms.iou_threshold, cfg.max_per_img)

            if len(det_bboxes) == 0:
                return []

            if rescale:
                assert img_meta.get('scale_factor') is not None
                det_bboxes /= det_bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))

            results = InstanceData(
                bboxes=det_bboxes,
                scores=scores * scores,
                labels=labels.int())
            result_list.append(results)

        return result_list

    def get_single_level_center_priors(self,
                                       batch_size,
                                       featmap_size,
                                       stride,
                                       dtype,
                                       device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype, device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype, device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def loss(self, x, data_samples):

        # gt_bboxes = [gt_instances.bboxes for gt_instances in batch_gt_instances]
        # gt_labels = [gt_instances.labels for gt_instances in batch_gt_instances]
        cls_score, bbox_pred = self(x)
        gts = unpack_gt_instances(data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = gts
        loss = self.loss_by_feat(cls_score, bbox_pred, batch_gt_instances, batch_img_metas,batch_gt_instances_ignore)
        return loss

    def loss_by_feat(self,
             cls_scores,
             bbox_preds,
             batch_gt_instances,
             batch_img_metas,
             cfg=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        """

        '''fake input for debug
        batch_gt_instances = InstanceData(
            labels=torch.tensor([7]).cuda(),
            bboxes=torch.tensor([[143.8004, 182.2277, 170.6345, 204.0171]]).cuda())
        import pdb;pdb.set_trace()
        '''
        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        cfg = self.test_cfg if cfg is None else cfg

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(
                num_imgs,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=flatten_cls_scores.device)
            for i, stride in enumerate(self.strides)]
        mlvl_center_priors = torch.cat(mlvl_priors_list, dim=1)

        device = cls_scores[0].device
        cls_scores = flatten_cls_scores
        bbox_preds = flatten_bbox_preds

        gt_bboxes = [gt.bboxes for gt in batch_gt_instances]
        gt_labels = [gt.labels for gt in batch_gt_instances]
        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_cls(
            cls_scores, (labels, label_scores), avg_factor=num_total_pos)

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False).squeeze(1)

        weight_targets = cls_scores.detach()
        weight_targets = weight_targets.max(dim=1)[0][pos_inds]
        norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
            )
            loss_dfl = self.loss_dfl(
                bbox_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
            )

        else:
            loss_bbox = bbox_preds.sum() * 0.0
            loss_dfl = bbox_preds.sum() * 0.0

        return dict(
            loss_cls=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
        )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
            self.get_target_single,
            mlvl_center_priors,
            cls_scores,
            bbox_preds,
            gt_bboxes_list,
            gt_labels_list,
        )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
                all_bbox_weights, all_dfl_targets, all_pos_num)

    def get_target_single(self,
                          center_priors,
                          cls_scores,
                          bbox_preds,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True,
                          gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        # assign gt and sample anchors
        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center,),
                                        self.num_classes,
                                        dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center, dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center, dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:
            return (labels, label_scores, label_weights,
                    bbox_targets, bbox_weights, dfl_targets, 0)
        pred_instances = InstanceData(
            bboxes=bbox_preds.detach(),
            scores=cls_scores.detach(),
            priors=center_priors)
        gt_instances = InstanceData(
            bboxes=gt_bboxes.detach(),
            labels=gt_labels.detach())
        

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances)

        pos_inds, neg_inds, pos_bbox_targets, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (
                    bbox2distance(center_priors[pos_inds, :2], pos_bbox_targets, self.reg_max)
                    / center_priors[pos_inds, None, 2]
            )
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets, bbox_weights,
                dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * quality_focal_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            use_sigmoid=self.use_sigmoid,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls

def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid # 8400, 81
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = func(pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss