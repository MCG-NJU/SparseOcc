import os
import mmcv
import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import inv
from mmcv.runner import get_dist_info
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from torchvision.transforms.functional import rotate


def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    # skip loading history frames
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        world_size = get_dist_info()[1]
        if world_size == 1 and self.test_mode:
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __init__(self, num_classes=18, inst_class_ids=[]):
        self.num_classes = num_classes
        self.inst_class_ids = inst_class_ids
    
    def __call__(self, results):
        occ_labels = np.load(results['occ_path'])
        semantics = occ_labels['semantics']  # [200, 200, 16]
        # mask_lidar = occ_labels['mask_lidar'].astype(np.bool_)  # [200, 200, 16]
        # mask_camera = occ_labels['mask_camera'].astype(np.bool_)  # [200, 200, 16]

        # results['mask_lidar'] = mask_lidar
        # results['mask_camera'] = mask_camera
  
        # instance GT
        if 'instances' in occ_labels.keys():
            instances = occ_labels['instances']
            instance_class_ids = [self.num_classes - 1]  # the 0-th class is always free class
            for i in range(1, instances.max() + 1):
                class_id = np.unique(semantics[instances == i])
                assert class_id.shape[0] == 1, "each instance must belong to only one class"
                instance_class_ids.append(class_id[0])
            instance_class_ids = np.array(instance_class_ids)
        else:
            instances = None
            instance_class_ids = None

        instance_count = 0
        final_instance_class_ids = []
        final_instances = np.ones_like(semantics) * 255  # empty space has instance id "255"

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics == class_id) == 0:
                continue

            if class_id in self.inst_class_ids:
                assert instances is not None, 'instance annotation not found'
                # treat as instances
                for instance_id in range(len(instance_class_ids)):
                    if instance_class_ids[instance_id] != class_id:
                        continue
                    final_instances[instances == instance_id] = instance_count
                    instance_count += 1
                    final_instance_class_ids.append(class_id)
            else:
                # treat as semantics
                final_instances[semantics == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)

        results['voxel_semantics'] = semantics
        results['voxel_instances'] = final_instances
        results['instance_class_ids'] = DC(to_tensor(final_instance_class_ids))

        if results.get('rotate_bda', False):
            semantics = torch.from_numpy(semantics).permute(2, 0, 1)  # [16, 200, 200]
            semantics = rotate(semantics, results['rotate_bda'], fill=255).permute(1, 2, 0)  # [200, 200, 16]
            results['voxel_semantics'] = semantics.numpy()

            final_instances = torch.from_numpy(final_instances).permute(2, 0, 1)  # [16, 200, 200]
            final_instances = rotate(final_instances, results['rotate_bda'], fill=255).permute(1, 2, 0)  # [200, 200, 16]
            results['voxel_instances'] = final_instances.numpy()

        if results.get('flip_dx', False):
            results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
            results['voxel_instances'] = results['voxel_instances'][::-1, ...].copy()
            
        if results.get('flip_dy', False):
            results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, ...].copy()
            results['voxel_instances'] = results['voxel_instances'][:, ::-1, ...].copy()

        return results


# https://github.com/HuangJunJie2017/BEVDet/blob/58c2587a8f89a1927926f0bdb6cde2917c91a9a5/mmdet3d/datasets/pipelines/loading.py#L1177
@PIPELINES.register_module()
class BEVAug(object):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        """
        Returns:
            rot_mat: (3, 3)
        """
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        
        return rot_mat

    def __call__(self, results):
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1

        # bda_rot: (3, 3)
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot

        results['bda_mat'] = bda_mat
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda

        for i in range(len(results['ego2lidar'])):
            results['ego2lidar'][i] = results['ego2lidar'][i] @ torch.inverse(bda_mat).numpy()  # [4, 4] @ [4, 4]

        return results
