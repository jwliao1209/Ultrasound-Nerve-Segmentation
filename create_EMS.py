from tqdm import tqdm
from torchvision.transforms import Resize
from libtiff import TIFF
from PIL import Image
import cv2
import os
import numpy as np


def MSK(image, erosion):
    interior = erosion
    interior[interior != 0] = 243
    boundary = image - interior
    boundary[boundary != 0] = 243
    new_Mask = interior + boundary

    return new_Mask


if __name__ == '__main__':
    load_path = r'\dataset\train'
    ker_size = 9
    iterations = 3
    ctr = []
    mask_path = os.path.join(load_path, 'train')
    TN_mask = [name for name in sorted(os.listdir(mask_path))
               if len(name.split('_')) == 3]
    kernel = np.ones((ker_size, ker_size), np.uint8)
    print('Wait for a miniute ...(about 1 to 2 mins)')

    if not os.path.isdir(os.path.join(load_path, 'train_mask')):
        os.mkdir(os.path.join(load_path, 'train_mask'))

    for name in tqdm(TN_mask):
        tif = TIFF.open(os.path.join(mask_path, name), mode='r')
        image = tif.read_image()
        tp = sum(sum(image != 0))
        ctr.append(tp)

        if tp == 0:
            image = Resize((448, 576))(Image.fromarray(image))
            image = np.array(image)
            image.dtype = 'uint8'
            cv2.imwrite(f'dataset/train_mask/{name}', image)

        else:
            image = Resize((448, 576))(Image.fromarray(image))
            image = np.array(image)
            erosion = cv2.erode(image, kernel, iterations=iterations)
            new_Mask = MSK(image, erosion)
            new_Mask.dtype = 'uint8'
            cv2.imwrite(f'dataset/train_mask/{name}', new_Mask)

    print('All new masks are created !')
