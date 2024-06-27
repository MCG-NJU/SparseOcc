import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES, build_loss
from mmdet.core import reduce_mean
from .utils import sparse2dense
from torch.cuda.amp import autocast
from torch.autograd import Variable


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


# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L48
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


# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py#L90
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
def mean(l, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    
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