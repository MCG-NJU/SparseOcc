_base_ = ['./r50_nuimg_704x256_8f.py']

occ_gt_root = 'data/nuscenes/occ3d_panoptic'

# For nuScenes we usually do 10-class detection
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

_num_frames_ = 8

model = dict(
    pts_bbox_head=dict(
        panoptic=True
    )
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names), inst_class_ids=[2, 3, 4, 5, 6, 7, 9, 10]),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names), inst_class_ids=[2, 3, 4, 5, 6, 7, 9, 10]),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar'))
]

data = dict(
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        occ_gt_root=occ_gt_root
    ),
    val=dict(
        pipeline=test_pipeline,
        occ_gt_root=occ_gt_root
    ),
    test=dict(
        pipeline=test_pipeline,
        occ_gt_root=occ_gt_root
    ),
)
