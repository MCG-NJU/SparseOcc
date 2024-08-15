import os
import glob
import mmcv
import numpy as np
import pkg_resources
from torch.utils.data import DataLoader
from .ray_metrics import main_rayiou
from .ego_pose_dataset import EgoPoseDataset

openocc_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

def evaluate_metrics(data_root, pred_dir, data_type):
    data_path = pkg_resources.resource_filename('rayiou_metrics', 'ego_infos_val.pkl')
    data_infos = mmcv.load(data_path)['infos']
    gt_filepaths = sorted(glob.glob(os.path.join(data_root, data_type, '*/*/*.npz')))

    # retrieve scene_name
    token2scene = {}
    for gt_path in gt_filepaths:
        token = gt_path.split('/')[-2]
        scene_name = gt_path.split('/')[-3]
        token2scene[token] = scene_name

    for i in range(len(data_infos)):
        scene_name = token2scene[data_infos[i]['token']]
        data_infos[i]['scene_name'] = scene_name

    lidar_origins = []
    occ_gts = []
    occ_preds = []

    for idx, batch in enumerate(DataLoader(EgoPoseDataset(data_infos), num_workers=8)):
        output_origin = batch[1]
        info = data_infos[idx]

        occ_path = os.path.join(data_root, data_type, info['scene_name'], info['token'], 'labels.npz')
        occ_gt = np.load(occ_path, allow_pickle=True)['semantics']
        occ_gt = np.reshape(occ_gt, [200, 200, 16]).astype(np.uint8)

        occ_path = os.path.join(pred_dir, info['token'] + '.npz')
        occ_pred = np.load(occ_path, allow_pickle=True)['pred']
        occ_pred = np.reshape(occ_pred, [200, 200, 16]).astype(np.uint8)
        
        lidar_origins.append(output_origin)
        occ_gts.append(occ_gt)
        occ_preds.append(occ_pred)
    
    if data_type == 'occ3d':
        occ_class_names = occ3d_class_names
    elif data_type == 'openocc_v2':
        occ_class_names = openocc_class_names
    else:
        raise ValueError("Invalid data_type. Support ['occ3d', 'openocc_v2']")
    
    metrics = main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)

    print('--- Evaluation Results ---')
    for k, v in metrics.items():
        print('%s: %.4f' % (k, v))

    return metrics
