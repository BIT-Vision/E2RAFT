import h5py
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import re

from .utils.EventToVoxel import events_to_voxel

class RealData():
    def __init__(self, data_path, name):
        # Load
        data_raw = h5py.File(os.path.join(data_path, name + '.h5'), 'r')
        events = data_raw['events']
        self.ps = events['ps']
        self.ts = events['ts']
        self.xs = events['xs']
        self.ys = events['ys']
        self.images = data_raw.get('images')

    def get_image(self, index):
        """
        :return image: (260,346,3)
        """
        image_key = 'image' + '{:0>9}'.format(index)
        return np.array(self.images[image_key])
    
    def get_events(self, index_start:int, index_end:int):
        """
        :returns events: [N,4] (x,y,t,p)
        """
        p = self.ps[index_start:index_end] * 2 - 1
        t = self.ts[index_start:index_end]
        x = self.xs[index_start:index_end]
        y = self.ys[index_start:index_end]
        return np.vstack((x, y, t, p)).transpose((1,0))

    def get_idx_imageToevent(self, index):
        image_key = 'image' + '{:0>9}'.format(index)
        return np.array(self.images[image_key].attrs['event_idx'])
    
    def len_images(self):
        return len(self.images)
    

class RealDataset(Dataset):
    def __init__(self, cfgs, name):
        # 初始化参数
        self.cfgs = cfgs
        self.voxel_bins = cfgs.voxel_bins
        self.crop_size = cfgs.crop_size_indoor if re.match(name, 'Indoor\S*') is None else cfgs.crop_size_outdoor

        self.realdata = RealData(cfgs.data_path, name)
        
        self.len = self.realdata.len_images() - 1
    
    def __getitem__(self, index):
        # 1. Images
        pre_image = self.realdata.get_image(index)
        next_image = self.realdata.get_image(index+1)

        # 2. Events
        index_start = self.realdata.get_idx_imageToevent(index)
        index_end = self.realdata.get_idx_imageToevent(index+1)
        events = self.realdata.get_events(index_start, index_end)
        # voxel = events_to_voxel(events, self.voxel_bins, pre_image.shape[1], pre_image.shape[2])
        voxel_pos = events_to_voxel(events, int(self.voxel_bins/2), pre_image.shape[0], pre_image.shape[1], pos=1) # Trans to Voxel
        voxel_pos = voxel_pos.transpose(1,2,0) / np.max(voxel_pos)
        voxel_neg = events_to_voxel(events, int(self.voxel_bins/2), pre_image.shape[0], pre_image.shape[1], pos=-1) * (-1) # Trans to Voxel
        voxel_neg = voxel_neg.transpose(1,2,0) / np.max(voxel_neg)
        voxel = np.concatenate((voxel_pos, voxel_neg), axis=2)

        # 3. Augment
        crop_height = self.crop_size[0]
        crop_width = self.crop_size[1]
        height = pre_image.shape[0]
        width = pre_image.shape[1]
        if height != crop_height or width != crop_width:
            assert crop_height <= height and crop_width <= width
            start_y = (height - crop_height) // 2
            start_x = (width - crop_width) // 2
            pre_image = pre_image[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
            next_image = next_image[start_y:start_y+crop_height, start_x:start_x+crop_width, :]
            voxel = voxel[start_y:start_y+crop_height, start_x:start_x+crop_width, :]

        # 4. ToTensor
        pre_image = torch.from_numpy(pre_image).permute(2,0,1).float()
        next_image = torch.from_numpy(next_image).permute(2,0,1).float()
        voxel = torch.from_numpy(voxel).permute(2,0,1).float()

        return pre_image, next_image, voxel

    def __len__(self):
        return self.len