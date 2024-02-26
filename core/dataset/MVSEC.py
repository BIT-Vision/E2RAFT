import os
import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils.Augmentor import Augmentor
from .utils.EventToVoxel import events_to_voxel
from .utils.mvsec import MVSEC

Valid_Index_ForTest = {
    'indoor_flying1': [314, 2199],
    'indoor_flying2': [314, 2199],
    'indoor_flying3': [314, 2199],
    'indoor_flying4': [196, 570],
    'outdoor_day1': [245, 3000],
    'outdoor_day2': [4375, 7002],
}
Valid_Index_ForTrain = {
    'outdoor_day2': [2100, 28400]
}


class MVSECData(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, cfgs, name='outdoor_day1', augmentor=True, split='test'):
        self.cfgs = cfgs
        self.name = name
        self.mvsec = MVSEC(os.path.join(cfgs.data_path, 'data_hdf5'), name)

        self.crop_size = cfgs.crop_size
        self.voxel_bins = cfgs.voxel_bins
        self.dt = cfgs.dt

        self.augmentor = Augmentor(cfgs.crop_size, do_flip=cfgs.do_flip) if augmentor else None
        if split == 'train':
            self.start_no = Valid_Index_ForTrain[name][0]
            self.data_len = Valid_Index_ForTrain[name][1] - Valid_Index_ForTrain[name][0] - cfgs.dt
        else:
            self.start_no = Valid_Index_ForTest[name][0]
            self.data_len = Valid_Index_ForTest[name][1] - Valid_Index_ForTest[name][0] - cfgs.dt

    def __getitem__(self, index):
        no = self.start_no + index
        # 0. Get Images
        pre_image = self.mvsec.get_image(no)
        next_image = self.mvsec.get_image(no + self.dt)
        # Dim changes: Gray to 3Gray
        if len(pre_image.shape) == 2:
            pre_image = np.tile(pre_image[..., None], (1, 1, 3))
            next_image = np.tile(next_image[..., None], (1, 1, 3))

        # 1. Get Events
        E_start = self.mvsec.get_idx_imageToevent(no)
        E_end = self.mvsec.get_idx_imageToevent(no + self.dt)
        events = self.mvsec.get_events(E_start, E_end)
        voxel_pos = events_to_voxel(events, int(self.voxel_bins/2), 260, 346, pos=1) # Trans to Voxel
        voxel_pos = voxel_pos.transpose(1,2,0) / np.max(voxel_pos) if np.max(voxel_pos)>0 else voxel_pos.transpose(1,2,0)
        voxel_neg = events_to_voxel(events, int(self.voxel_bins/2), 260, 346, pos=-1) * (-1) # Trans to Voxel
        voxel_neg = voxel_neg.transpose(1,2,0) / np.max(voxel_neg) if np.max(voxel_neg)>0 else voxel_neg.transpose(1,2,0)
        voxel = np.concatenate((voxel_pos, voxel_neg), axis=2)

        # 2. Get Flow
        T_start = self.mvsec.get_time_ofimage(no)
        T_end = self.mvsec.get_time_ofimage(no + self.dt)
        flow = self.mvsec.estimate_flow(T_start, T_end)

        # 3. Get Valid
        valid = np.logical_and(np.linalg.norm(x=flow, ord=2, axis=2, keepdims=False) > 0,
                               np.logical_and(np.absolute(flow[..., 0]) < 1000, 
                                              np.absolute(flow[..., 1]) < 1000)).astype(np.float64)
        if self.name == 'outdoor_day1' or self.name == 'outdoor_day2':
            valid[193:,:]=False

        # 4. Augment
        if self.augmentor is not None:
            pre_image, next_image, voxel, flow,  valid = self.augmentor(pre_image, next_image, voxel, flow,  valid)
        else:
            crop_height = self.crop_size[0]
            crop_width = self.crop_size[1]
            height = voxel.shape[0]
            width = voxel.shape[1]
            if height != crop_height or width != crop_width:
                assert crop_height <= height and crop_width <= width
                # 中心裁剪
                start_y = (height - crop_height) // 2
                start_x = (width - crop_width) // 2
                pre_image = pre_image[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                next_image = next_image[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                voxel = voxel[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                flow = flow[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
                valid = valid[start_y:start_y+crop_height, start_x:start_x+crop_width]

        # 5. To Tensor
        pre_image = torch.from_numpy(pre_image).permute(2,0,1).float()
        next_image = torch.from_numpy(next_image).permute(2,0,1).float()
        voxel = torch.from_numpy(voxel).permute(2,0,1).float()
        flow = torch.from_numpy(flow).permute(2,0,1).float()
        valid = torch.from_numpy(valid).float()

        return pre_image, next_image, voxel, flow, valid

    def __len__(self):
        return self.data_len
