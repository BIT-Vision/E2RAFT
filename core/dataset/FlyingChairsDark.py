import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.Augmentor import Augmentor
from .utils.EventToVoxel import events_to_voxel
from .utils.FlyingChairs2_IO import read


class FlyingChairsData(Dataset):
    def __init__(self, cfgs, split='train', augmentor=False):
        # 初始化参数
        self.cfgs = cfgs
        self.split = split

        self.augmentor = Augmentor(cfgs.crop_size, do_flip=cfgs.do_flip) if augmentor else None

        self.IF_dark = cfgs.IF_dark
        self.voxel_bins = cfgs.voxel_bins

        self.data_path = ''
        self.eventH5_path = ''
        if split == 'train':
            self.data_path = os.path.join(cfgs.data_path, 'train')
            self.eventH5_path = os.path.join(cfgs.data_path, 'voxels_train_b5_pn.hdf5')
        else:
            self.data_path = os.path.join(cfgs.data_path, 'val')
            self.eventH5_path = os.path.join(cfgs.data_path, 'voxels_val_b5_pn.hdf5')

        # Data
        if self.IF_dark:
            self.images_path_0_dark = sorted(glob.glob(os.path.join(self.data_path, 'dark', '*img_0.npy')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))
            self.images_path_1_dark = sorted(glob.glob(os.path.join(self.data_path, 'dark', '*img_1.npy')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))
        else:
            self.images_path_0 = sorted(glob.glob(os.path.join(self.data_path, '*img_0.png')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))
            self.images_path_1 = sorted(glob.glob(os.path.join(self.data_path, '*img_1.png')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))
        self.flows_path = sorted(glob.glob(os.path.join(self.data_path, '*flow_01.flo')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))
        self.event_data = h5py.File(self.eventH5_path, 'r')
            
        self.len = len(self.event_data)

    def __getitem__(self, index):
        # Get Images
        train_ratio = 1
        if self.IF_dark and (self.split != 'train' or np.random.rand() < train_ratio):
            pre_image = np.load(self.images_path_0_dark[index])
            next_image = np.load(self.images_path_1_dark[index])
        else:
            pre_image = read(self.images_path_0[index])
            next_image = read(self.images_path_1[index])

        # Get Events
        voxel = self.event_data[str(index)][:]
        
        # Get GroundTruth-Flow
        gt_flow = read(self.flows_path[index])

        # Get Valid
        valid = np.logical_and(np.logical_or(np.absolute(gt_flow[..., 0]) > 0, np.absolute(gt_flow[..., 1]) > 0),
                               np.logical_and(np.absolute(gt_flow[..., 0]) < 400, np.absolute(gt_flow[..., 1]) < 400))
        valid = valid.astype(np.float64)

        # Augmentor
        if self.augmentor is not None:
            pre_image, next_image, voxel, gt_flow, valid = self.augmentor(pre_image, next_image, voxel, gt_flow, valid)

        # To Tensor
        pre_image = torch.from_numpy(pre_image).permute(2, 0, 1).float()
        next_image = torch.from_numpy(next_image).permute(2, 0, 1).float()
        voxel = torch.from_numpy(voxel).permute(2, 0, 1).float()
        gt_flow = torch.from_numpy(gt_flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()

        # img   : (c,h,w)
        # flow  : (c,h,w)
        # voxel : (c,h,w)
        # valid : (h,w)
        return pre_image, next_image, voxel, gt_flow, valid

    def __len__(self):
        # 22232 pairs image
        return self.len
