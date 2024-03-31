import os
import numpy as np
from sklearn.neighbors import KDTree
from termcolor import colored
from functools import reduce
from typing import Iterable

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        if num_classes == 18:
            self.class_names = [
                'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation','free'
            ]
        elif num_classes == 2:
            self.class_names = ['non-free', 'free']
        
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        #return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        result = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        result[hist.sum(1) == 0] = float('nan')
        return result

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        if self.num_classes == 2:
            masked_semantics_pred = np.copy(masked_semantics_pred)
            masked_semantics_gt = np.copy(masked_semantics_gt)
            masked_semantics_pred[masked_semantics_pred < 17] = 0
            masked_semantics_pred[masked_semantics_pred == 17] = 1
            masked_semantics_gt[masked_semantics_gt < 17] = 0
            masked_semantics_gt[masked_semantics_gt == 17] = 1
        
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))

        return round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)


class Metric_FScore():
    def __init__(self,
                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt=0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8

    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera ):
        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy=0
            completeness=0
            fmean=0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy+self.eps) + 1 / (completeness+self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self,):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))
        return self.tot_f1_mean / self.cnt

class Metric_mRecall():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 pred_classes=2,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        if num_classes == 18:
            self.class_names = [
                'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation','free'
            ]
        elif num_classes == 2:
            self.class_names = ['non-free', 'free']
        
        self.pred_classes = pred_classes
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.pred_classes))   # n_cl, p_cl
        self.cnt = 0

    def hist_info(self, n_cl, p_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                p_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl * p_cl
            ).reshape(n_cl, p_cl),   # 18, 2
            correct,
            labeled,
        )

    def per_class_recall(self, hist):
        return hist[:, 1] / hist.sum(1)   ## recall 

    def compute_mRecall(self, pred, label, n_classes, p_classes):
        hist = np.zeros((n_classes, p_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, p_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mRecalls = self.per_class_recall(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mRecalls) * 100, 2), hist

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        if self.pred_classes == 2:
            masked_semantics_pred = np.copy(masked_semantics_pred)
            masked_semantics_gt = np.copy(masked_semantics_gt)
            masked_semantics_pred[masked_semantics_pred < 17] = 1  
            masked_semantics_pred[masked_semantics_pred == 17] = 0 # 0 is free

        _, _hist = self.compute_mRecall(masked_semantics_pred, masked_semantics_gt, self.num_classes, self.pred_classes)
        self.hist += _hist

    def count_mrecall(self):
        mRecall = self.per_class_recall(self.hist)
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class Recall of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - Recall = ' + str(round(mRecall[ind_class] * 100, 2)))

        print(f'===> mRecall of {self.cnt} samples: ' + str(round(np.nanmean(mRecall[:self.num_classes-1]) * 100, 2)))

        return round(np.nanmean(mRecall[:self.num_classes-1]) * 100, 2)


# modified from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/evaluation/functional/panoptic_seg_eval.py#L10
class Metric_Panoptic():
    def __init__(self, 
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ignore_index: Iterable[int]=[],
                 ):
        """
        Args:
            ignore_index (llist): Class ids that not be considered in pq counting.
        """
        if num_classes == 18:
            self.class_names = [
                'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation','free'
            ]
        else:
            raise ValueError
        
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.ignore_index = ignore_index
        self.id_offset = 2 ** 16
        self.eps = 1e-5
        
        self.min_num_points = 20
        self.include = np.array(
            [n for n in range(self.num_classes - 1) if n not in self.ignore_index],
            dtype=int)
        self.cnt = 0
        
        # panoptic stuff
        self.pan_tp = np.zeros(self.num_classes, dtype=int)
        self.pan_iou = np.zeros(self.num_classes, dtype=np.double)
        self.pan_fp = np.zeros(self.num_classes, dtype=int)
        self.pan_fn = np.zeros(self.num_classes, dtype=int)
        
    def add_batch(self,semantics_pred,semantics_gt,instances_pred,instances_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
            masked_instances_gt = instances_gt[mask_camera]
            masked_instances_pred = instances_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
            masked_instances_gt = instances_gt[mask_lidar]
            masked_instances_pred = instances_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
            masked_instances_gt = instances_gt
            masked_instances_pred = instances_pred
        self.add_panoptic_sample(masked_semantics_pred, masked_semantics_gt, masked_instances_pred, masked_instances_gt) 
    
    def add_panoptic_sample(self, semantics_pred, semantics_gt, instances_pred, instances_gt):
        """Add one sample of panoptic predictions and ground truths for
        evaluation.

        Args:
            semantics_pred (np.ndarray): Semantic predictions.
            semantics_gt (np.ndarray): Semantic ground truths.
            instances_pred (np.ndarray): Instance predictions.
            instances_gt (np.ndarray): Instance ground truths.
        """
        # get instance_class_id from instance_gt
        instance_class_ids = [self.num_classes - 1]
        for i in range(1, instances_gt.max() + 1):
            class_id = np.unique(semantics_gt[instances_gt == i])
            # assert class_id.shape[0] == 1, "each instance must belong to only one class"
            if class_id.shape[0] == 1:
                instance_class_ids.append(class_id[0])
            else:
                instance_class_ids.append(self.num_classes - 1)
        instance_class_ids = np.array(instance_class_ids)

        instance_count = 1
        final_instance_class_ids = []
        final_instances = np.zeros_like(instances_gt)  # empty space has instance id "0"

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics_gt == class_id) == 0:
                continue

            if self.class_names[class_id] in ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']:
                # treat as instances
                for instance_id in range(len(instance_class_ids)):
                    if instance_class_ids[instance_id] != class_id:
                        continue
                    final_instances[instances_gt == instance_id] = instance_count
                    instance_count += 1
                    final_instance_class_ids.append(class_id)
            else:
                # treat as semantics
                final_instances[semantics_gt == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)
                
        instances_gt = final_instances
        
        # avoid zero (ignored label)
        instances_pred = instances_pred + 1
        instances_gt = instances_gt + 1
        
        for cl in self.ignore_index:
            # make a mask for this class
            gt_not_in_excl_mask = semantics_gt != cl
            # remove all other points
            semantics_pred = semantics_pred[gt_not_in_excl_mask]
            semantics_gt = semantics_gt[gt_not_in_excl_mask]
            instances_pred = instances_pred[gt_not_in_excl_mask]
            instances_gt = instances_gt[gt_not_in_excl_mask]
        
        # for each class (except the ignored ones)
        for cl in self.include:
            # get a class mask
            pred_inst_in_cl_mask = semantics_pred == cl
            gt_inst_in_cl_mask = semantics_gt == cl

            # get instance points in class (makes outside stuff 0)
            pred_inst_in_cl = instances_pred * pred_inst_in_cl_mask.astype(int)
            gt_inst_in_cl = instances_gt * gt_inst_in_cl_mask.astype(int)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                pred_inst_in_cl[pred_inst_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                gt_inst_in_cl[gt_inst_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])

            # generate intersection using offset
            valid_combos = np.logical_and(pred_inst_in_cl > 0,
                                          gt_inst_in_cl > 0)
            id_offset_combo = pred_inst_in_cl[
                valid_combos] + self.id_offset * gt_inst_in_cl[valid_combos]
            unique_combo, counts_combo = np.unique(
                id_offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.id_offset
            pred_labels = unique_combo % self.id_offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array(
                [counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(float) / unions.astype(float)

            tp_indexes = ious > 0.5
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id]
                          for id in pred_labels[tp_indexes]]] = True

            # count the FN
            if len(counts_gt) > 0:
                self.pan_fn[cl] += np.sum(
                    np.logical_and(counts_gt >= self.min_num_points,
                                   ~matched_gt))

            # count the FP
            if len(matched_pred) > 0:
                self.pan_fp[cl] += np.sum(
                    np.logical_and(counts_pred >= self.min_num_points,
                                   ~matched_pred))
    
    def count_pq(self, ):
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double)
            + 0.5 * self.pan_fn.astype(np.double), self.eps)
        pq_all = sq_all * rq_all
        
        # mask classes not occurring in dataset
        mask = (self.pan_tp + self.pan_fp + self.pan_fn) > 0
        sq_all[~mask] = float('nan')
        rq_all[~mask] = float('nan')
        pq_all[~mask] = float('nan')
        
        # then do the REAL mean (no ignored classes)
        sq = round(np.nanmean(sq_all[self.include]) * 100, 2)
        rq = round(np.nanmean(rq_all[self.include]) * 100, 2)
        pq = round(np.nanmean(pq_all[self.include]) * 100, 2)
        
        print(f'===> per class sq, rq, pq of {self.cnt} samples:')
        for ind_class in self.include:
            print(f'===> {self.class_names[ind_class]} -' + \
                  f' sq = {round(sq_all[ind_class] * 100, 2)},' + \
                  f' rq = {round(rq_all[ind_class] * 100, 2)},' + \
                  f' pq = {round(pq_all[ind_class] * 100, 2)}')
        
        print(f'===> sq of {self.cnt} samples: ' + str(sq))
        print(f'===> rq of {self.cnt} samples: ' + str(rq))
        print(f'===> pq of {self.cnt} samples: ' + str(pq))

        return (pq, sq, rq)