import os
import tqdm
import glob
import pickle
import argparse
import numpy as np
import torch
import multiprocessing
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box


parser = argparse.ArgumentParser()
parser.add_argument('--nusc-root', default='data/nuscenes')
parser.add_argument('--occ3d-root', default='data/nuscenes/occ3d')
parser.add_argument('--output-dir', default='data/nuscenes/occ3d_panoptic')
parser.add_argument('--version', default='v1.0-trainval')
args = parser.parse_args()

token2path = {}
for gt_path in glob.glob(os.path.join(args.occ3d_root, '*/*/*.npz')):
    token = gt_path.split('/')[-2]
    token2path[token] = gt_path

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

det_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


def convert_to_nusc_box(bboxes, lift_center=False, wlh_margin=0.0):
    results = []
    for q in range(bboxes.shape[0]):

        bbox = bboxes[q].copy()
        if lift_center:
            bbox[2] += bbox[5] * 0.5

        bbox_yaw = -bbox[6] - np.pi / 2
        orientation = Quaternion(axis=[0, 0, 1], radians=bbox_yaw).inverse

        box = Box(
            center=[bbox[0], bbox[1], bbox[2]],
            # 0.8 in pc range is roungly 2 voxels in occ grid
            # enlarge bbox to include voxels on the edge
            size=[bbox[3]+wlh_margin, bbox[4]+wlh_margin, bbox[5]+wlh_margin],
            orientation=orientation,
        )

        results.append(box)

    return results


def meshgrid3d(occ_size, pc_range):  # points in ego coord
    W, H, D = occ_size
    
    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def process_add_instance_info(sample):
    point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
    occ_size = [200, 200, 16]
    num_classes = 18
    
    occ_gt_path = token2path[sample['token']]
    occ_labels = np.load(occ_gt_path)
    
    occ_gt = occ_labels['semantics']
    gt_boxes = sample['gt_boxes']
    gt_names = sample['gt_names']
    
    bboxes = convert_to_nusc_box(gt_boxes)
    
    instance_gt = np.zeros(occ_gt.shape).astype(np.uint8)
    instance_id = 1
    
    pts = meshgrid3d(occ_size, point_cloud_range).numpy()
    
    # filter out free voxels to accelerate
    valid_idx = np.where(occ_gt < num_classes - 1)
    flatten_occ_gt = occ_gt[valid_idx]
    flatten_inst_gt = instance_gt[valid_idx]
    flatten_pts = pts[valid_idx]
    
    instance_boxes = []
    instance_class_ids = []
    
    for i in range(len(gt_names)):
        if gt_names[i] not in occ_class_names:
            continue
        occ_tag_id = occ_class_names.index(gt_names[i])
            
        # Move box to ego vehicle coord system
        bbox = bboxes[i]
        bbox.rotate(Quaternion(sample['lidar2ego_rotation']))
        bbox.translate(np.array(sample['lidar2ego_translation']))
        
        mask = points_in_box(bbox, flatten_pts.transpose(1, 0))
        
        # ignore voxels not belonging to this class
        mask[mask] = (flatten_occ_gt[mask] == occ_tag_id)
        # ignore voxels already occupied
        mask[mask] = (flatten_inst_gt[mask] == 0)
        
        # only instance with at least 1 voxel will be recorded
        if mask.sum() > 0:
            flatten_inst_gt[mask] = instance_id
            instance_id += 1
            
            # enlarge boxes to include voxels on the edge
            new_box = bbox.copy()
            new_box.wlh = new_box.wlh + 0.8
            
            instance_boxes.append(new_box)
            instance_class_ids.append(occ_tag_id)
    
    # classes that should be viewed as one instance
    all_class_ids_unique = np.unique(occ_gt)
    for i, class_name in enumerate(occ_class_names):
        if class_name in det_class_names or class_name == 'free' or i not in all_class_ids_unique:
            continue
        flatten_inst_gt[flatten_occ_gt == i] = instance_id
        instance_id += 1
    
    # post process unconvered non-occupied voxels
    uncover_idx = np.where(flatten_inst_gt == 0)
    uncover_pts = flatten_pts[uncover_idx]
    uncover_inst_gt = np.zeros_like(uncover_pts[..., 0]).astype(np.uint8)
    unconver_occ_gt = flatten_occ_gt[uncover_idx]
    
    # uncover_inst_dist records the dist between each voxel and its current nearest bbox's center
    uncover_inst_dist = np.ones_like(uncover_pts[..., 0]) * 1e8
    for i, box in enumerate(instance_boxes):
        # important, non-background inst id starts from 1
        inst_id = i + 1
        class_id = instance_class_ids[i]
        mask = points_in_box(box, uncover_pts.transpose(1, 0))
        # mask voxels not belonging to this class
        mask[unconver_occ_gt != class_id] = False
        dist = np.sum((box.center - uncover_pts) ** 2, axis=-1)
        # voxels that have already been assigned to a closer box's instance should be ignored
        # voxels that not inside the box should be ignored
        # `mask[(dist >= uncover_inst_dist)]=False` is right, as it only transforms True masks into False without converting False into True
        # to give readers a more clear understanding, the most standard writing is `mask[mask & (dist >= uncover_inst_dist)]=False`
        mask[dist >= uncover_inst_dist] = False
        # mask[mask & (dist >= uncover_inst_dist)]=False
        
        # important: only voxels inside the box (mask = True) and having no closer identical-class box need to update dist
        uncover_inst_dist[mask] = dist[mask]
        uncover_inst_gt[mask] = inst_id
        
    flatten_inst_gt[uncover_idx] = uncover_inst_gt
    
    instance_gt[valid_idx] = flatten_inst_gt
    # not using this checking function yet
    # assert (instance_gt == 0).sum() - (occ_gt == num_classes-1).sum() < 100, "too many non-free voxels are not assigned to any instance in %s"%(occ_gt_path)
    # global max_margin
    # if max_margin < (instance_gt == 0).sum() - (occ_gt == num_classes-1).sum():
    #     print("###### new max margin: ", max(max_margin, (instance_gt == 0).sum() - (occ_gt == num_classes-1).sum()))
    # max_margin = max(max_margin, (instance_gt == 0).sum() - (occ_gt == num_classes-1).sum())
    
    # save to original path
    data_split = occ_gt_path.split(os.path.sep)[-3:]
    data_path = os.path.sep.join(data_split)
    
    ##### Warning: Using args.xxx (global variable) here is strongly unrecommended
    save_path = os.path.join(args.output_dir, data_path)
    
    save_dir = os.path.split(save_path)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if np.unique(instance_gt).shape[0] != instance_gt.max()+1:
        print('warning: some instance masks are covered by following ones %s'%(save_dir))
    
    # only semantic and mask information is needed to be reserved
    retain_keys = ['semantics', 'mask_lidar', 'mask_camera']   
    new_occ_labels = {k: occ_labels[k] for k in retain_keys}
    new_occ_labels['instances'] = instance_gt
    np.savez_compressed(save_path, **new_occ_labels)


def add_instance_info(sample_infos):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # all cpus participate in multi processing
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    with tqdm.tqdm(total=len(sample_infos['infos'])) as pbar:
        for _ in pool.imap(process_add_instance_info, sample_infos['infos']):
            pbar.update(1)
    
    pool.close()
    pool.join()


if __name__ == '__main__':
    if args.version == 'v1.0-trainval':
        sample_infos = pickle.load(open(os.path.join(args.nusc_root, 'nuscenes_infos_train_sweep.pkl'), 'rb'))
        add_instance_info(sample_infos)

        sample_infos = pickle.load(open(os.path.join(args.nusc_root, 'nuscenes_infos_val_sweep.pkl'), 'rb'))
        add_instance_info(sample_infos)

    elif args.version == 'v1.0-test':
        sample_infos = pickle.load(open(os.path.join(args.nusc_root, 'nuscenes_infos_test_sweep.pkl'), 'rb'))
        add_instance_info(sample_infos)

    else:
        raise ValueError
