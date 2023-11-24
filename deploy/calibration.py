import _init_paths
import os
import torch
import argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from lib.utils.utils import load_yaml
from lib.dataset.dataset import LightTrackMDataset as data_builder


def parse_args():
    parser = argparse.ArgumentParser(description='Generate LightTrackM Calibration Data')
    parser.add_argument('--cfg', type=str, default='../experiments/LightTrack.yaml', help='yaml configure file name')
    args = parser.parse_args()
    args.calib_z = "./calibration/template/"
    args.calib_x = "./calibration/search/"
    args.calib_path = [args.calib_z, args.calib_x]
    args.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    args.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return args


def calibration():
    # preprocess and configure
    args = parse_args()
    config = edict(load_yaml(args.cfg))
    for calib_path in args.calib_path:
        if not os.path.exists(calib_path): os.makedirs(calib_path)

    # build dataset
    train_set = data_builder(config, calibration=True)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8,
                              pin_memory=True, sampler=None, drop_last=True)

    for iter_id, batch_data in enumerate(train_loader):
        template = batch_data['template']  # bx3x128x128
        search = batch_data['search']  # bx3x256x256

        print('template shape: ', template.shape)
        print('search shape: ', search.shape)

        if iter_id < 128:
            # z = np.transpose(template.squeeze(0).numpy().astype(np.int8), (1, 2, 0))
            z = template.numpy().astype(np.uint8)
            z.tofile(args.calib_path[0] + "z" + "_" + str(iter_id) + ".bin")
            # x = np.transpose(search.squeeze(0).numpy().astype(np.int8), (1, 2, 0))
            x = search.numpy().astype(np.uint8)
            x.tofile(args.calib_path[1] + "x" + "_" + str(iter_id) + ".bin")
        else:
            break


if __name__ == '__main__':
    calibration()
