import os
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from loaders.old_metrics import Metric_mIoU


def main(args):
    pred_filepaths = sorted(glob.glob(os.path.join(args.pred_dir, '*.npz')))
    gt_filepaths = sorted(glob.glob(os.path.join(args.data_root, 'occ3d', '*/*/*.npz')))

    eval_metrics_miou = Metric_mIoU(
        num_classes=18,
        use_lidar_mask=False,
        use_image_mask=True)

    for pred_filepath in tqdm(pred_filepaths):
        sample_token = os.path.basename(pred_filepath).split('.')[0]
        for gt_filepath in gt_filepaths:
            if sample_token in gt_filepath:
                sem_pred = np.load(pred_filepath, allow_pickle=True)['pred']
                sem_pred = np.reshape(sem_pred, [200, 200, 16])
                occ_gt = np.load(gt_filepath, allow_pickle=True)

                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
                
                eval_metrics_miou.add_batch(sem_pred, gt_semantics, mask_lidar, mask_camera)

    eval_metrics_miou.count_miou()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default='data/nuscenes')
    parser.add_argument("--pred-dir", type=str)
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)
