import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ColorJitter
import torch

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Augmentor:
    def __init__(self, crop_size, do_flip=True):
        # crop augmentation params
        self.crop_size = crop_size

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if torch.FloatTensor(1).uniform_(0, 1).item() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(
                Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(
                Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(
                Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def spatial_transform(self, img1, img2, voxel, flow, valid):
        # flip
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                voxel = voxel[:, ::-1]
                valid = valid[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                voxel = voxel[::-1, :]
                valid = valid[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # random crop
        y0 = 0 if img1.shape[0] == self.crop_size[0] else np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = 0 if img1.shape[1] == self.crop_size[1] else np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        voxel = voxel[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, voxel, flow, valid

    def __call__(self, img1, img2, voxel, flow, valid):
        # img   : (h,w,c)
        # flow  : (h,w,c)
        # voxel : (h,w,c)
        # valid : (h,w)
        img1, img2, voxel, flow, valid = self.spatial_transform(img1, img2, voxel, flow, valid)
        img1, img2 = self.color_transform(img1, img2)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        voxel = np.ascontiguousarray(voxel)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, voxel, flow, valid
