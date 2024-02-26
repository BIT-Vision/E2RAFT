import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1  # 归一化到(-1,1)

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def initialize_flow(img):
    """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
    coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

    # optical flow computed as difference: flow = coords1 - coords0
    return coords0, coords1


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1

def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def upsample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = flow.shape
    mask = mask.view(N, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3, 3], padding=1)
    up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 2, 8 * H, 8 * W)






import numpy
import torch
from torch import nn
from torch.nn.functional import grid_sample
from scipy.spatial import transform
from scipy import interpolate
from matplotlib import pyplot as plt


def grid_sample_values(input, height, width):
    # ================================ Grid Sample Values ============================= #
    # Input:    Torch Tensor [3,H*W]m where the 3 Dimensions mean [x,y,z]               #
    # Height:   Image Height                                                            #
    # Width:    Image Width                                                             #
    # --------------------------------------------------------------------------------- #
    # Output:   tuple(value_ipl, valid_mask)                                            #
    #               value_ipl       -> [H,W]: Interpolated values                       #
    #               valid_mask      -> [H,W]: 1: Point is valid, 0: Point is invalid    #
    # ================================================================================= #
    device = input.device
    ceil = torch.stack([torch.ceil(input[0,:]), torch.ceil(input[1,:]), input[2,:]])
    floor = torch.stack([torch.floor(input[0,:]), torch.floor(input[1,:]), input[2,:]])
    z = input[2,:].clone()

    values_ipl = torch.zeros(height*width, device=device)
    weights_acc = torch.zeros(height*width, device=device)
    # Iterate over all ceil/floor points
    for x_vals in [floor[0], ceil[0]]:
        for y_vals in [floor[1], ceil[1]]:
            # Mask Points that are in the image
            in_bounds_mask = (x_vals < width) & (x_vals >=0) & (y_vals < height) & (y_vals >= 0)

            # Calculate weights, according to their real distance to the floored/ceiled value
            weights = (1 - (input[0]-x_vals).abs()) * (1 - (input[1]-y_vals).abs())

            # Put them into the right grid
            indices = (x_vals + width * y_vals).long()
            values_ipl.put_(indices[in_bounds_mask], (z * weights)[in_bounds_mask], accumulate=True)
            weights_acc.put_(indices[in_bounds_mask], weights[in_bounds_mask], accumulate=True)

    # Mask of valid pixels -> Everywhere where we have an interpolated value
    valid_mask = weights_acc.clone()
    valid_mask[valid_mask > 0] = 1
    valid_mask= valid_mask.bool().reshape([height,width])

    # Divide by weights to get interpolated values
    values_ipl = values_ipl / (weights_acc + 1e-15)
    values_rs = values_ipl.reshape([height,width])

    return values_rs.unsqueeze(0).clone(), valid_mask.unsqueeze(0).clone()

def forward_interpolate_pytorch(flow_in):
    # Same as the numpy implementation, but differentiable :)
    # Flow: [B,2,H,W]
    flow = flow_in.clone()
    if len(flow.shape) < 4:
        flow = flow.unsqueeze(0)

    b, _, h, w = flow.shape
    device = flow.device

    dx ,dy = flow[:,0], flow[:,1]
    y0, x0 = torch.meshgrid(torch.arange(0, h, 1), torch.arange(0, w, 1))
    x0 = torch.stack([x0]*b).to(device)
    y0 = torch.stack([y0]*b).to(device)

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.flatten(start_dim=1)
    y1 = y1.flatten(start_dim=1)
    dx = dx.flatten(start_dim=1)
    dy = dy.flatten(start_dim=1)

    # Interpolate Griddata...
    # Note that a Nearest Neighbor Interpolation would be better. But there does not exist a pytorch fcn yet.
    # See issue: https://github.com/pytorch/pytorch/issues/50339
    flow_new = torch.zeros(flow.shape, device=device)
    for i in range(b):
        flow_new[i,0] = grid_sample_values(torch.stack([x1[i],y1[i],dx[i]]), h, w)[0]
        flow_new[i,1] = grid_sample_values(torch.stack([x1[i],y1[i],dy[i]]), h, w)[0]

    return flow_new

class ImagePadder(object):
    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        height, width = image.shape[-2:]
        if self.pad_width is None:
            self.pad_height = (self.min_size - height % self.min_size)%self.min_size
            self.pad_width = (self.min_size - width % self.min_size)%self.min_size
        else:
            pad_height = (self.min_size - height % self.min_size)%self.min_size
            pad_width = (self.min_size - width % self.min_size)%self.min_size
            if pad_height != self.pad_height or pad_width != self.pad_width:
                raise
        return nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]
