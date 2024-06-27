import copy
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import FFN
from .sparsebev_transformer import AdaptiveMixing
from .utils import DUMP
from .checkpoint import checkpoint as cp
from .sparsebev_sampling import sampling_4d, make_sample_points_from_mask
from .sparse_voxel_decoder import SparseVoxelDecoder


@TRANSFORMER.register_module()
class SparseOccTransformer(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_queries=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None,
                 topk_training=None,
                 topk_testing=None):
        super().__init__()
        self.num_frames = num_frames
        
        self.voxel_decoder = SparseVoxelDecoder(
            embed_dims=embed_dims,
            num_layers=3,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            semantic=True,
            topk_training=topk_training,
            topk_testing=topk_testing
        )
        self.decoder = MaskFormerOccDecoder(
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_frames=num_frames,
            num_queries=num_queries,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size,
        )
        
    @torch.no_grad()
    def init_weights(self):
        self.voxel_decoder.init_weights()
        self.decoder.init_weights()

    def forward(self, mlvl_feats, img_metas):
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = 6, TN // 6, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
            feat = feat.reshape(B*T*G, N, H, W, C)  # [BTG, N, H, W, C]
            mlvl_feats[lvl] = feat.contiguous()
        
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(feat.device)  # [B, N, 4, 4]
        ego2lidar = np.asarray([m['ego2lidar'] for m in img_metas]).astype(np.float32)
        ego2lidar = torch.from_numpy(ego2lidar).to(feat.device)  # [B, N, 4, 4]
        
        img_metas = copy.deepcopy(img_metas)
        img_metas[0]['lidar2img'] = torch.matmul(lidar2img, ego2lidar)

        occ_preds = self.voxel_decoder(mlvl_feats, img_metas=img_metas)
        mask_preds, class_preds = self.decoder(occ_preds, mlvl_feats, img_metas)
        
        return occ_preds, mask_preds, class_preds


class MaskFormerOccDecoder(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_queries=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None):
        super().__init__()

        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_frames = num_frames

        self.decoder_layer = MaskFormerOccDecoderLayer(
            embed_dims=embed_dims,
            mask_dim=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size,
        )
        
        self.query_feat = nn.Embedding(num_queries, embed_dims)
        self.query_pos = nn.Embedding(num_queries, embed_dims)
        
    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()
        
    def forward(self, occ_preds, mlvl_feats, img_metas):
        occ_loc, occ_pred, _, mask_feat, _ = occ_preds[-1]
        bs = mask_feat.shape[0]
        query_feat = self.query_feat.weight[None].repeat(bs, 1, 1)
        query_pos = self.query_pos.weight[None].repeat(bs, 1, 1)
        
        valid_map, mask_pred, class_pred = self.decoder_layer.pred_segmentation(query_feat, mask_feat)
        
        class_preds = [class_pred]
        mask_preds = [mask_pred]

        for i in range(self.num_layers):
            DUMP.stage_count = i
            query_feat, valid_map, mask_pred, class_pred = self.decoder_layer(
                query_feat, valid_map, mask_pred, occ_preds, mlvl_feats, query_pos, img_metas
            )
            mask_preds.append(mask_pred)
            class_preds.append(class_pred)

        return mask_preds, class_preds


class MaskFormerOccDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 mask_dim=None,
                 num_frames=None,
                 num_queries=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None):
        super().__init__()
        
        self.pc_range = pc_range
        self.occ_size = occ_size
        
        self.self_attn = MaskFormerSelfAttention(embed_dims, num_heads=8)
        self.sampling = MaskFormerSampling(embed_dims, num_frames, num_groups, num_points, num_levels, pc_range=pc_range, occ_size=occ_size)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=num_groups, out_points=128)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)
        self.mask_proj = nn.Linear(embed_dims, mask_dim)
        self.classifier = nn.Linear(embed_dims, num_classes - 1)
        
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()
        
    def forward(self, query_feat, valid_map, mask_pred, occ_preds, mlvl_feats, query_pos, img_metas):
        """
        query_feat: [bs, num_query, embed_dim]
        valid_map: [bs, num_query, num_voxel]
        mask_pred: [bs, num_query, num_voxel]
        occ_preds: list(occ_loc, occ_pred, _, mask_feat, scale), all voxel decoder's outputs
            mask_feat: [bs, num_voxel, embed_dim]
            occ_pred: [bs, num_voxel]
            occ_loc: [bs, num_voxel, 3]
        """
        occ_loc, occ_pred, _, mask_feat, _ = occ_preds[-1]
        query_feat = self.norm1(self.self_attn(query_feat, query_pos=query_pos))

        sampled_feat = self.sampling(query_feat, valid_map, occ_loc, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        
        query_feat = self.norm3(self.ffn(query_feat))
        
        valid_map, mask_pred, class_pred = self.pred_segmentation(query_feat, mask_feat)
        return query_feat, valid_map, mask_pred, class_pred
    
    def pred_segmentation(self, query_feat, mask_feat):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_pred_segmentation, query_feat, mask_feat, use_reentrant=False)
        else:
            return self.inner_pred_segmentation(query_feat, mask_feat)
    
    def inner_pred_segmentation(self, query_feat, mask_feat):
        class_pred = self.classifier(query_feat)
        feat_proj = self.mask_proj(query_feat)
        mask_pred = torch.einsum("bqc,bnc->bqn", feat_proj, mask_feat)
        valid_map = (mask_pred > 0.0)
        
        return valid_map, mask_pred, class_pred


class MaskFormerSelfAttention(BaseModule):
    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos
                
    def inner_forward(self, query, mask = None, key_padding_mask = None,query_pos= None):
        q = k = self.with_pos_embed(query, query_pos)
        tgt = self.self_attn(q, k, value=query, attn_mask=mask, key_padding_mask=key_padding_mask)[0]
        query = query + self.dropout(tgt)
        return query

    def forward(self, query, mask = None, key_padding_mask = None,query_pos= None):
        if self.training and query.requires_grad:
            return cp(self.inner_forward, query, mask, key_padding_mask, query_pos, use_reentrant=False)
        else:
            return self.inner_forward(query, mask, key_padding_mask, query_pos)


class MaskFormerSampling(BaseModule):
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, pc_range=[], occ_size=[], init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.occ_size = occ_size

        self.offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)
        
    def init_weights(self, ):
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def inner_forward(self, query_feat, valid_map, occ_loc, mlvl_feats, img_metas):
        '''
        valid_map: [B, Q, W, H, D]
        query_feat: [B, Q, C]
        '''
        B, Q = query_feat.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # sampling offset of all frames
        offset = self.offset(query_feat).view(B, Q, self.num_groups * self.num_points, 3)  # [B, Q, GP, 3]
        sampling_points = make_sample_points_from_mask(valid_map, self.pc_range, self.occ_size, self.num_groups*self.num_points, occ_loc, offset)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w
        )  # [B, Q, G, FP, C]

        return sampled_feats

    def forward(self, query_feat, valid_map, occ_loc,  mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_feat, valid_map, occ_loc, mlvl_feats, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_feat, valid_map, occ_loc, mlvl_feats, img_metas)
