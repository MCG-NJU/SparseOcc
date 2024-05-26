import os
import cv2
import utils
import logging
import argparse
import importlib
import torch
import numpy as np
from tqdm import tqdm
from mmcv import Config, DictAction
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from configs.r50_nuimg_704x256_8f import point_cloud_range as pc_range
from configs.r50_nuimg_704x256_8f import occ_size
from configs.r50_nuimg_704x256_8f import occ_class_names
import random
import colorsys
from typing import List, Tuple
import functools
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


@functools.lru_cache(50)
def get_evenly_distributed_colors(
    count: int,
) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    # lru cache caches color tuples
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    random.shuffle(HSV_tuples)
    return list(
        map(
            lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(
                np.uint8
            ),
            HSV_tuples,
        )
    )


color_10 = np.vstack(get_evenly_distributed_colors(10)).tolist()
color_6 = np.vstack(get_evenly_distributed_colors(6)).tolist()
color_45 = np.vstack(get_evenly_distributed_colors(45)).tolist()
color_4 = np.vstack(get_evenly_distributed_colors(4)).tolist()
color_12 = np.vstack(get_evenly_distributed_colors(12)).tolist()
color_32 = np.vstack(get_evenly_distributed_colors(32)).tolist()
color_3 = np.vstack(get_evenly_distributed_colors(3)).tolist()
color_8 = np.vstack(get_evenly_distributed_colors(8)).tolist()

color_map = [
    [[0, 0, 0, ]],    # others
    [[255, 120, 50, ]],  # barrier              orangey
    color_10,  #10 @bicycle              pink
    color_6,  #(6) @bus                  yellow
    color_45,  #45 @car                  blue
    color_4,  #(4) @construction_vehicle cyan
    color_12,  #12 @motorcycle           dark orange
    color_32,  #32 @pedestrian           red
    [[255, 240, 150, ]],  # traffic_cone         light yellow
    color_3,  #(3) @trailer              brown
    color_8,  #8 @truck                purple
    [[214, 215, 214]],  # driveable_surface    dark pink
    [[175,   0,  75, ]],       # other_flat           dark red  
    [[75, 0, 75, ]],  # sidewalk             dard purple
    [[150, 240, 80, ]],  # terrain              light green
    [[230, 230, 250, ]],  # manmade              white
    [[100, 200, 50, ]],  # vegetation           green
]

color_map = np.array(  # color map for semantic occupancy
    [
        [0, 0, 0, ],    # others
        [255, 120, 50, ],  # barrier              orangey
        [255, 192, 203, ],  # bicycle              pink
        [255, 255, 0, ],  # bus                  yellow
        [0, 150, 245, ],  # car                  blue
        [0, 255, 255, ],  # construction_vehicle cyan
        [200, 180, 0, ],  # motorcycle           dark orange
        [255, 0, 0, ],  # pedestrian           red
        [255, 240, 150, ],  # traffic_cone         light yellow
        [135, 60, 0, ],  # trailer              brown
        [160, 32, 240, ],  # truck                purple
        [255, 0, 255, ],  # driveable_surface    dark pink
        [175,   0,  75, ],       # other_flat           dark red
        # [139, 137, 137, ],   
        [75, 0, 75, ],  # sidewalk             dard purple
        [150, 240, 80, ],  # terrain              light green
        [230, 230, 250, ],  # manmade              white
        [0, 175, 0, ],  # vegetation           green
        [255, 255, 255,] # free
    ]
)

def occ2img(semantics):
    H, W, D = semantics.shape

    free_id = len(occ_class_names) - 1
    semantics_2d = np.ones([H, W], dtype=np.int32) * free_id

    for i in range(D):
        semantics_i = semantics[..., i]
        non_free_mask = (semantics_i != free_id)
        semantics_2d[non_free_mask] = semantics_i[non_free_mask]

    viz = color_map[semantics_2d]
    viz = viz[..., :3]
    #viz = cv2.resize(viz, dsize=(800, 800))

    return viz

def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--viz-dir', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # use val-mini for visualization
    #cfgs.data.val.ann_file = cfgs.data.val.ann_file.replace('val', 'val_mini')

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need one GPU
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

    # logging
    utils.init_logging(None, cfgs.debug)
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))

    # random seed
    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])
    model.eval()

    logging.info('Loading checkpoint from %s' % args.weights)
    load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    for i, data in tqdm(enumerate(val_loader)):

        #print(data['img_metas'].data[0][0]['filename'][:6])

        with torch.no_grad():
            occ_pred = model(return_loss=False, rescale=True, **data)[0]

            sem_pred = torch.from_numpy(occ_pred['sem_pred'])[0]  # [N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))[0]  # [N, 3]
            
            # sparse to dense
            free_id = len(occ_class_names) - 1
            dense_pred = torch.ones(occ_size, device=sem_pred.device, dtype=sem_pred.dtype) * free_id  # [200, 200, 16]
            dense_pred[occ_loc[..., 0], occ_loc[..., 1], occ_loc[..., 2]] = sem_pred
            
            sem_pred = dense_pred.numpy()

            cv2.imwrite(os.path.join(args.viz_dir, 'sem_%04d.jpg' % i), occ2img(sem_pred)[..., ::-1])


if __name__ == '__main__':
    main()
