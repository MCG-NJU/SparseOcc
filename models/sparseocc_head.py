import numpy as np
import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.builder import build_loss
from mmdet.models.utils import build_transformer
from .matcher import HungarianMatcher
from .loss_utils import CE_ssc_loss, lovasz_softmax, get_voxel_decoder_loss_input


NUSC_CLASS_FREQ = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027, 2074468, 413451, 2384460,
    5916653, 175883646, 4275424, 51393615, 61411620, 105975596, 116424404, 1892500630
])


@HEADS.register_module()
class SparseOccHead(nn.Module): 
    def __init__(self,
                 transformer=None,
                 class_names=None,
                 embed_dims=None,
                 occ_size=None,
                 pc_range=None,
                 loss_cfgs=None,
                 panoptic=False,
                 **kwargs):
        super(SparseOccHead, self).__init__()
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.pc_range = pc_range
        self.occ_size = occ_size
        self.embed_dims = embed_dims
        self.score_threshold = 0.3
        self.overlap_threshold = 0.8
        self.panoptic = panoptic

        self.transformer = build_transformer(transformer)
        self.criterions = {k: build_loss(loss_cfg) for k, loss_cfg in loss_cfgs.items()}
        self.matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)

        self.class_weights = torch.from_numpy(1 / np.log(NUSC_CLASS_FREQ + 0.001))

    def init_weights(self):
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        occ_preds, mask_preds, class_preds = self.transformer(mlvl_feats, img_metas=img_metas)
        
        return {
            'occ_preds': occ_preds, 
            'mask_preds': mask_preds, 
            'class_preds': class_preds
        }

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, voxel_instances, instance_class_ids, preds_dicts, mask_camera=None):
        return self.loss_single(voxel_semantics, voxel_instances, instance_class_ids, preds_dicts, mask_camera)

    def loss_single(self, voxel_semantics, voxel_instances, instance_class_ids, preds_dicts, mask_camera=None):
        loss_dict = {}
        B = voxel_instances.shape[0]

        if mask_camera is not None:
            assert mask_camera.shape == voxel_semantics.shape
            assert mask_camera.dtype == torch.bool
        
        for i, (occ_loc_i, _, seg_pred_i, _, scale) in enumerate(preds_dicts['occ_preds']):
            loss_dict_i = {}
            for b in range(B):
                loss_dict_i_b = {}
                seg_pred_i_sparse, voxel_semantics_sparse, sparse_mask = get_voxel_decoder_loss_input(
                    voxel_semantics[b:b + 1],
                    occ_loc_i[b:b + 1],
                    seg_pred_i[b:b + 1] if seg_pred_i is not None else None,
                    scale,
                    self.num_classes
                )

                loss_dict_i_b['loss_sem_lovasz'] = lovasz_softmax(torch.softmax(seg_pred_i_sparse, dim=1), voxel_semantics_sparse)

                valid_mask = (voxel_semantics_sparse < 255)
                seg_pred_i_sparse = seg_pred_i_sparse[valid_mask].transpose(0, 1).unsqueeze(0)  # [K, CLS] -> [B, CLS, K]
                voxel_semantics_sparse = voxel_semantics_sparse[valid_mask].unsqueeze(0)  # [K] -> [B, K]

                if 'loss_geo_scal' in self.criterions.keys():
                    loss_dict_i_b['loss_geo_scal'] = self.criterions['loss_geo_scal'](seg_pred_i_sparse, voxel_semantics_sparse)  
                if 'loss_sem_scal' in self.criterions.keys():
                    loss_dict_i_b['loss_sem_scal'] = self.criterions['loss_sem_scal'](seg_pred_i_sparse, voxel_semantics_sparse)

                loss_dict_i_b['loss_sem_ce'] = CE_ssc_loss(seg_pred_i_sparse, voxel_semantics_sparse, self.class_weights.type_as(seg_pred_i_sparse))

                for loss_key in loss_dict_i_b.keys():
                    loss_dict_i[loss_key] = loss_dict_i.get(loss_key, 0) + loss_dict_i_b[loss_key] / B

            for k, v in loss_dict_i.items():
                loss_dict['%s_%d' % (k, i)] = v

        occ_loc = preds_dicts['occ_preds'][-1][0]
        
        batch_idx = torch.arange(B)[:, None, None].expand(B, occ_loc.shape[1], 1).to(occ_loc.device)
        occ_loc = occ_loc.reshape(-1, 3)
        voxel_instances = voxel_instances[batch_idx.reshape(-1), occ_loc[..., 0], occ_loc[..., 1], occ_loc[..., 2]]
        voxel_instances = voxel_instances.reshape(B, -1)  # [B, N]

        if mask_camera is not None:
            mask_camera = mask_camera[batch_idx.reshape(-1), occ_loc[..., 0], occ_loc[..., 1], occ_loc[..., 2]]
            mask_camera = mask_camera.reshape(B, -1)  # [B, N]
        
        # drop instances if it has no positive voxels
        for b in range(B):
            instance_count = instance_class_ids[b].shape[0]
            instance_voxel_counts = torch.bincount(voxel_instances[b].long())  # [255]
            id_map = torch.cumsum(instance_voxel_counts > 0, dim=0) - 1
            id_map[255] = 255  # empty space still has an id of 255
            voxel_instances[b] = id_map[voxel_instances[b].long()]
            instance_class_ids[b] = instance_class_ids[b][instance_voxel_counts[:instance_count] > 0]

        for i, pred in enumerate(preds_dicts['mask_preds']):
            indices = self.matcher(pred, preds_dicts['class_preds'][i], voxel_instances, instance_class_ids, mask_camera)
            loss_mask, loss_dice, loss_class = self.criterions['loss_mask2former'](
                pred, preds_dicts['class_preds'][i], voxel_instances, instance_class_ids, indices, mask_camera)
            loss_dict['loss_mask_{:d}'.format(i)] = loss_mask
            loss_dict['loss_dice_mask_{:d}'.format(i)] = loss_dice
            loss_dict['loss_class_{:d}'.format(i)] = loss_class

        return loss_dict
    
    def merge_occ_pred(self, outs):
        mask_cls = outs['class_preds'][-1].sigmoid()
        mask_pred = outs['mask_preds'][-1].sigmoid()
        occ_indices = outs['occ_preds'][-1][0]
        
        sem_pred = self.merge_semseg(mask_cls, mask_pred)  # [B, C, N]
        outs['sem_pred'] = sem_pred
        outs['occ_loc'] = occ_indices

        if self.panoptic:
            pano_inst, pano_sem = self.merge_panoseg(mask_cls, mask_pred)  # [B, C, N]
            outs['pano_inst'] = pano_inst
            outs['pano_sem'] = pano_sem
        
        return outs
    
    # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/mask_former_model.py#L242
    def merge_semseg(self, mask_cls, mask_pred):
        valid_mask = mask_cls.max(dim=-1).values > self.score_threshold
        mask_cls[~valid_mask] = 0.0

        semseg = torch.einsum("bqc,bqn->bcn", mask_cls, mask_pred)
        if semseg.shape[1] == self.num_classes:
            semseg = semseg[:, :-1]
        
        cls_score, cls_id = torch.max(semseg, dim=1)
        cls_id[cls_score < 0.01] = self.num_classes - 1
        return cls_id  # [B, N]
    
    def merge_panoseg(self, mask_cls, mask_pred):
        pano_inst, pano_sem = [], []
        for b in range(mask_cls.shape[0]):
            pano_inst_b, pano_sem_b = self.merge_panoseg_single(
                mask_cls[b:b + 1],
                mask_pred[b:b + 1]
            )
            pano_inst.append(pano_inst_b)
            pano_sem.append(pano_sem_b)
        
        pano_inst = torch.cat(pano_inst, dim=0)
        pano_sem = torch.cat(pano_sem, dim=0)
        
        return pano_inst, pano_sem

    # https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py#L286
    def merge_panoseg_single(self, mask_cls, mask_pred):
        assert mask_cls.shape[0] == 1, "bs != 1"
        scores, labels = mask_cls.max(-1)
        
        # filter out low score and background instances
        keep = labels.ne(self.num_classes - 1) & (scores > self.score_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1) * cur_masks

        N = cur_masks.shape[-1]
        instance_seg = torch.zeros((N), dtype=torch.int32, device=cur_masks.device)
        semantic_seg = torch.ones((N), dtype=torch.int32, device=cur_masks.device) * (self.num_classes - 1)
        
        current_segment_id = 0
        stuff_memory_list = {self.num_classes - 1: 0}

        # skip all process if no mask is detected
        if cur_masks.shape[0] != 0:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # [N]
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()

                # moving objects are treated as instances
                is_thing = self.class_names[pred_class] in [
                    'car', 'truck', 'construction_vehicle', 'bus',
                    'trailer', 'motorcycle', 'bicycle', 'pedestrian'
                ]

                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not is_thing:
                        if int(pred_class) in stuff_memory_list.keys():
                            instance_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    instance_seg[mask] = current_segment_id
                    semantic_seg[mask] = pred_class
        
        instance_seg = instance_seg.unsqueeze(0)
        semantic_seg = semantic_seg.unsqueeze(0)
        
        return instance_seg, semantic_seg  # [B, N]
