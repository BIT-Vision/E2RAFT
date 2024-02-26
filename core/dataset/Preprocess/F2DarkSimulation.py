import glob
import os.path
import numpy as np
from skimage.exposure import exposure
from tqdm import tqdm

from FlyingChairs2_IO import read

def add_noise(img1, img2):
    g = 38.25
    p = 19.50/255.
    a, b = abs(np.random.normal(0,p)), abs(np.random.normal(0,g))
    sd1, sd2 = abs(a*img1+b), abs(a*img2+b)
    noisy_img1 = img1 + np.random.normal(0, sd1)
    noisy_img2 = img2 + np.random.normal(0, sd2)
    return np.clip(noisy_img1.astype(int), 0, 255).astype(np.uint8), np.clip(noisy_img2.astype(int), 0, 255).astype(np.uint8)

def image_darken(pre_image, next_image):
    dark_gamma = np.random.uniform(2, 6)
    pre_image = exposure.adjust_gamma(pre_image, dark_gamma).astype(np.uint8)
    next_image = exposure.adjust_gamma(next_image, dark_gamma).astype(np.uint8)
    pre_image, next_image = add_noise(pre_image, next_image)

    return pre_image, next_image

if __name__ == '__main__':
    np.random.seed(0)
    
    # Input and Output
    data_path = "/Datasets/FlyingChairs2/train/"
    save_path = os.path.join(data_path, 'dark')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Load Images
    pre_images_path = sorted(glob.glob(os.path.join(data_path, '*img_0.png')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))
    next_images_path = sorted(glob.glob(os.path.join(data_path, '*img_1.png')), key=lambda x: int(str(x).split('-')[0].split(os.path.sep)[-1]))

    # Darkness Simulation
    for index in tqdm(range(len(pre_images_path))):
        if not os.path.exists(os.path.join(save_path, pre_images_path[index].split(os.sep)[-1].split('.')[0]+'.npy')):
            pre_image = read(pre_images_path[index])
            next_image = read(next_images_path[index])
            pre_image, next_image = image_darken(pre_image, next_image)

            np.save(os.path.join(save_path, pre_images_path[index].split(os.sep)[-1].split('.')[0]+'.npy'), pre_image)
            np.save(os.path.join(save_path, next_images_path[index].split(os.sep)[-1].split('.')[0]+'.npy'), next_image)
