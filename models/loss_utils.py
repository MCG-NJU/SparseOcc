import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES, build_loss
from mmdet.core import reduce_mean
from .utils import sparse2dense
import numpy as np
from torch.cuda.amp import autocast
from torch.autograd import Variable
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

nusc_class_frequencies = np.array([
 944004,
 1897170,
 152386,
 2391677,
 16957802,
 724139,
 189027,
 2074468,
 413451,
 2384460,
 5916653,
 175883646,
 4275424,
 51393615,
 61411620,
 105975596,
 116424404,
 1892500630
 ])

def calc_voxel_decoder_loss(voxel_semantics, occ_loc_i, occ_pred_i, seg_pred_i, scale, num_classes=18, mask_camera=None):
    loss_dict = dict()
    assert voxel_semantics.shape[0] == 1  # bs = 1
    voxel_semantics = voxel_semantics.long()

    occ_pred_dense, sparse_mask = sparse2dense(
        occ_loc_i, occ_pred_i,
        dense_shape=[200 // scale, 200 // scale, 16 // scale]
    )
    occ_pred_dense = F.interpolate(occ_pred_dense[:, None], scale_factor=scale)[:, 0]  # [B, W, H, D]
    sparse_mask = F.interpolate(sparse_mask[:, None].float(), scale_factor=scale)[:, 0].bool()

    loss_dict['loss_occ'] = compute_occ_loss(occ_pred_dense, voxel_semantics, sparse_mask, num_classes=num_classes, mask_camera=mask_camera)

    if seg_pred_i is not None:  # semantic prediction
        assert seg_pred_i.shape[-1] == num_classes-1
        
        seg_pred_dense, _ = sparse2dense(
            occ_loc_i, seg_pred_i,
            dense_shape=[200 // scale, 200 // scale, 16 // scale, num_classes-1],
            empty_value=torch.zeros((num_classes-1)).to(seg_pred_i)
        )
        seg_pred_dense = seg_pred_dense.permute(0, 4, 1, 2, 3)   # [B, CLS, W, H, D]
        seg_pred_dense = F.interpolate(seg_pred_dense, scale_factor=scale)
        seg_pred_dense = seg_pred_dense.permute(0, 2, 3, 4, 1)   # [B, W, H, D, CLS]

        seg_pred_i_sparse = seg_pred_dense[sparse_mask]
        voxel_semantics_sparse = voxel_semantics[sparse_mask]

        non_free_mask = (voxel_semantics_sparse != num_classes-1)
        seg_pred_i_sparse = seg_pred_i_sparse[non_free_mask]
        voxel_semantics_sparse = voxel_semantics_sparse[non_free_mask]

        loss_dict['loss_sem'] = F.cross_entropy(seg_pred_i_sparse, voxel_semantics_sparse)

    return loss_dict

def get_voxel_decoder_loss_input(voxel_semantics, occ_loc_i, seg_pred_i, scale, num_classes=18):
    assert voxel_semantics.shape[0] == 1  # bs = 1
    voxel_semantics = voxel_semantics.long()

    if seg_pred_i is not None:  # semantic prediction
        assert seg_pred_i.shape[-1] == num_classes
        
        seg_pred_dense, sparse_mask = sparse2dense(
            occ_loc_i, seg_pred_i,
            dense_shape=[200 // scale, 200 // scale, 16 // scale, num_classes],
            empty_value=torch.zeros((num_classes)).to(seg_pred_i)
        )
        sparse_mask = F.interpolate(sparse_mask[:, None].float(), scale_factor=scale)[:, 0].bool()
        seg_pred_dense = seg_pred_dense.permute(0, 4, 1, 2, 3)   # [B, CLS, W, H, D]
        seg_pred_dense = F.interpolate(seg_pred_dense, scale_factor=scale)
        seg_pred_dense = seg_pred_dense.permute(0, 2, 3, 4, 1)   # [B, W, H, D, CLS]

        seg_pred_i_sparse = seg_pred_dense[sparse_mask]  # [K, CLS]
        voxel_semantics_sparse = voxel_semantics[sparse_mask]  # [K]

    return seg_pred_i_sparse, voxel_semantics_sparse, sparse_mask



def compute_occ_loss(occ_pred, occ_target, mask=None, pos_weight=1.0, num_classes=18, mask_camera=None):  # 2-cls occupancy pred
    '''
    :param occ_pred: (Tensor), predicted occupancy, (N)
    :param occ_target: (Tensor), ground truth occupancy, (N)
    :param mask: (Bool Tensor), mask, (N)
    '''
    occ_pred = occ_pred.view(-1)
    occ_target = occ_target.view(-1)
    assert occ_pred.shape[0] == occ_target.shape[0]
    
    if mask is not None:
        assert mask.dtype == torch.bool
        mask = mask.view(-1)
        occ_pred = occ_pred[mask]
        occ_target = occ_target[mask]

    # balance class distribution by assigning different weights to each class
    cls_count = torch.bincount(occ_target)  # [18]
    cls_weight = cls_count.sum().float() / cls_count
    cls_weight[:-1] *= pos_weight
    cls_weight = torch.index_select(cls_weight, 0, occ_target.long())
    
    # 2-cls recon
    free_id = num_classes - 1
    occ_target = occ_target.clone()
    occ_target[occ_target < free_id] = 1
    occ_target[occ_target == free_id] = 0

    return F.binary_cross_entropy_with_logits(occ_pred, occ_target.float(), weight=cls_weight)


def compute_scal_loss(pred, gt, class_id, reverse=False, ignore_index=255):
    p = pred[:, class_id, :]
    completion_target = (gt == class_id).long()
    
    loss = torch.zeros(pred.shape[0], device=pred.device)
    
    if reverse:
        p = 1 - p
        completion_target = ((gt != class_id) & (gt != ignore_index)).long()
    
    target_sum = completion_target.sum(dim=(1))
    mask = (target_sum > 0)
    
    p = p[torch.where(mask)]
    completion_target = completion_target[torch.where(mask)]
    
    nominator = torch.sum(p * completion_target, dim=(1))
    
    p_mask = torch.where(torch.sum(p, dim=(1)) > 0)
    if p_mask[0].shape[0] > 0:
        precision = nominator[p_mask] / torch.sum(p[p_mask], dim=(1))
        loss_precision = F.binary_cross_entropy(
            precision, torch.ones_like(precision),
            reduction='none'
        )
        loss[torch.where(mask)[0][p_mask]] += loss_precision
        
    t_mask = torch.where(torch.sum(completion_target, dim=(1)) > 0)
    if t_mask[0].shape[0] > 0:
        recall = nominator[t_mask] / torch.sum(completion_target[t_mask], dim=(1))
        loss_recall = F.binary_cross_entropy(
            recall, torch.ones_like(recall),
            reduction='none'
        )
        loss[torch.where(mask)[0][t_mask]] += loss_recall
        
    ct_mask = torch.where(torch.sum(1 - completion_target, dim=(1)) > 0)
    if ct_mask[0].shape[0] > 0:
        specificity = torch.sum((1 - p[ct_mask]) * (1 - completion_target[ct_mask]), dim=(1)) / (
            torch.sum(1 - completion_target[ct_mask], dim=(1))
        )
        loss_ct = F.binary_cross_entropy(
            specificity, torch.ones_like(specificity),
            reduction='none'
        )
        loss[torch.where(mask)[0][ct_mask]] += loss_ct
        
    return loss, mask


@LOSSES.register_module()
class GeoScalLoss(nn.Module):
    def __init__(self, 
                 num_classes,
                 loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        
    def forward(self, pred, gt):
        loss = torch.tensor(0, device=pred.device, dtype=pred.dtype)
        pred = F.softmax(pred, dim=1)
        
        loss, _ = compute_scal_loss(pred, gt, self.num_classes - 1, reverse=True)
        return self.loss_weight * torch.mean(loss)


@LOSSES.register_module()
class SemScalLoss(nn.Module):
    def __init__(self, 
                 num_classes,
                 class_weights=None,
                 loss_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        if self.class_weights is not None:
            assert len(self.class_weights) == self.num_classes, "number of class weights must equal to class number"
        else:
            self.class_weights = [1.0 for _ in range(self.num_classes)]
        self.loss_weight = loss_weight
        
    def forward(self, pred, gt):
        pred = F.softmax(pred, dim=1)
        batch_size = pred.shape[0]
        loss = torch.zeros(batch_size, device=pred.device)
        count = torch.zeros(batch_size, device=pred.device)
        for i in range(self.num_classes):
            loss_cls, mask_cls = compute_scal_loss(pred, gt, i)
            count += mask_cls.long()
            loss += loss_cls * self.class_weights[i]
        
        return self.loss_weight * (loss / count).mean()


# borrowed from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L21
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        mask_camera: torch.Tensor
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if mask_camera is not None:
        inputs = inputs[:, :, mask_camera]
        targets = targets[:, :, mask_camera]
    
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.squeeze(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


# borrowed from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L48
def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        mask_camera: torch.Tensor
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    # [M, 1, K]
    if mask_camera is not None:
        mask_camera = mask_camera.to(torch.int32)
        mask_camera = mask_camera[None, None, ...].expand(targets.shape[0], 1, mask_camera.shape[-1])
        loss = F.binary_cross_entropy_with_logits(inputs, targets, mask_camera, reduction="none")
    else:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    return loss.mean(2).mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
    """
    :param: prediction: the predicted tensor, must be [BS, C, ...]
    """

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=ignore_index, reduction="mean"
    )
    # from IPython import embed
    # embed()
    # exit()
    with autocast(False):
        loss = criterion(pred, target.long())

    return loss

# https://github.com/NVlabs/FB-BEV/blob/832bd81866823a913a4c69552e1ca61ae34ac211/mmdet3d/models/fbbev/modules/occ_loss_utils/lovasz_softmax.py#L22
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

# https://github.com/NVlabs/FB-BEV/blob/832bd81866823a913a4c69552e1ca61ae34ac211/mmdet3d/models/fbbev/modules/occ_loss_utils/lovasz_softmax.py#L157
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        with autocast(False):
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss

# https://github.com/NVlabs/FB-BEV/blob/832bd81866823a913a4c69552e1ca61ae34ac211/mmdet3d/models/fbbev/modules/occ_loss_utils/lovasz_softmax.py#L176
def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

# https://github.com/NVlabs/FB-BEV/blob/832bd81866823a913a4c69552e1ca61ae34ac211/mmdet3d/models/fbbev/modules/occ_loss_utils/lovasz_softmax.py#L207
def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 2:
        if ignore is not None:
            valid = (labels != ignore)
            probas = probas[valid]
            labels = labels[valid]
        return probas, labels

    elif probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight

    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.
    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class CustomFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=100.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(CustomFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        H, W = 200, 200

        xy, yx = torch.meshgrid([torch.arange(H)-H/2,  torch.arange(W)-W/2])
        c = torch.stack([xy,yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c/c_max + 1).cuda()


    def forward(self,
                pred,
                target,
                sparse_mask=None,
                weight=None,
                avg_factor=None,
                ignore_index=255,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        B, H, W, D = sparse_mask.shape
        c = self.c[None, :, :, None].repeat(B, 1, 1, D)
        if sparse_mask is not None:
            c = c[sparse_mask]
        else:
            c = c.reshape(-1)

        visible_mask = (target!=ignore_index).reshape(-1).nonzero().squeeze(-1)
        weight_mask = weight[None,:] * c[visible_mask, None]
        # visible_mask[:, None]

        num_classes = pred.size(1)
        if sparse_mask is not None:
            pred = pred.transpose(1, 2).reshape(-1, num_classes)[visible_mask]
        else:
            pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]

        target = target.reshape(-1)[visible_mask]

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss


            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target.to(torch.long),
                weight_mask,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls

# modified from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L90
@LOSSES.register_module()
class Mask2FormerLoss(nn.Module):
    def __init__(self, 
                 num_classes,
                 loss_cls_weight=1.0, 
                 loss_mask_weight=1.0, 
                 loss_dice_weight=1.0, 
                 no_class_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        self.no_class_weight = no_class_weight
        self.empty_weight = torch.ones(self.num_classes)
        self.empty_weight[-1] = self.no_class_weight
        self.loss_cls = build_loss(dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0
        ))
        
    def forward(self, mask_pred, class_pred, mask_gt, class_gt, indices, mask_camera):
        bs = mask_pred.shape[0]
        loss_masks = torch.tensor(0).to(mask_pred)
        loss_dices = torch.tensor(0).to(mask_pred)
        loss_classes = torch.tensor(0).to(mask_pred)

        num_total_pos = sum([tc.numel() for tc in class_gt])
        avg_factor = torch.clamp(reduce_mean(class_pred.new_tensor([num_total_pos * 1.0])), min=1).item()
        
        for b in range(bs):
            mask_camera_b = mask_camera[b] if mask_camera is not None else None# N
            tgt_mask = mask_gt[b]
            num_instances = class_gt[b].shape[0]

            tgt_class = class_gt[b]
            tgt_mask = (tgt_mask.unsqueeze(-1) == torch.arange(num_instances).to(mask_gt.device))
            tgt_mask = tgt_mask.permute(1, 0)
            
            src_idx, tgt_idx = indices[b]
            src_mask = mask_pred[b][src_idx]   # [M, N], M is number of gt instances, N is number of remaining voxels
            tgt_mask = tgt_mask[tgt_idx]   # [M, N]
            src_class = class_pred[b]   # [Q, CLS]
            
            # pad non-aligned queries' tgt classes with 'no class'
            pad_tgt_class = torch.full(
                (src_class.shape[0], ), self.num_classes - 1, dtype=torch.int64, device=class_pred.device
            )   # [Q]
            pad_tgt_class[src_idx] = tgt_class[tgt_idx]
            
            # only calculates loss mask for aligned pairs
            loss_mask, loss_dice = self.loss_masks(src_mask, tgt_mask, avg_factor=avg_factor, mask_camera=mask_camera_b)
            # calculates loss class for all queries
            loss_class = self.loss_labels(src_class, pad_tgt_class, self.empty_weight.to(src_class.device), avg_factor=avg_factor)
            
            loss_masks += loss_mask * self.loss_mask_weight
            loss_dices += loss_dice * self.loss_dice_weight
            loss_classes += loss_class * self.loss_cls_weight
            
        return loss_masks, loss_dices, loss_classes
    
    # mask2former use point sampling to calculate loss of fewer important points
    # we omit point sampling as we have limited number of points
    def loss_masks(self, src_mask, tgt_mask, avg_factor=None, mask_camera=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        num_masks = tgt_mask.shape[0]
        src_mask = src_mask.view(num_masks, 1, -1)
        tgt_mask = tgt_mask.view(num_masks, 1, -1)
        
        if avg_factor is None:
            avg_factor = num_masks

        loss_dice = dice_loss(src_mask, tgt_mask, avg_factor, mask_camera)
        loss_mask = sigmoid_ce_loss(src_mask, tgt_mask.float(), avg_factor, mask_camera)
        
        return loss_mask, loss_dice
        
    def loss_labels(self, src_class, tgt_class, empty_weight=None, avg_factor=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        return self.loss_cls(
            src_class, tgt_class, torch.ones_like(tgt_class), avg_factor=avg_factor
        ).mean()

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n