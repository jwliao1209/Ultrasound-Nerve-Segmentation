import os
import glob
import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from src.utils import *
from src.transforms import *


def get_image_name(path):
    return int(path.split('/')[-1][:-4])


class BPDataset(Dataset):
    def __init__(self, root, SUBJECT, IMG, transform=None, training=True):
        '''
        Inputs:
        root (str): path pointing to '/dataset/train/'
        SUBJECT (list[str]): list of {subject}
        IMG (list[str]): list of {img}
        transform (Transform): callable function that perform preprocessing

        we will read image and label with the following file names
        image path: 'root/{subject}_{img}.tif'
        label path: 'root/{subject}_{img}_mask.tif'

        Outputs:
        image (torch.tensor): tensor of shape (3, H, W), torch.float()
        label (torch.tensor): tensor of shape (1, H, W), torch.float()
        '''
        self.root = root
        self.SUBJECT = SUBJECT
        self.IMG = IMG
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.IMG)

    def __getitem__(self, idx):
        if self.training:
            sub = self.SUBJECT[idx]
            img = self.IMG[idx]
            data = {
                'image': os.path.join(
                    self.root, 'train', f"{sub}_{img}.tif"),
                'label': os.path.join(
                    self.root, 'train', f"{sub}_{img}_mask.tif"),
                'pixelnum': 0
            }

        else:
            data = {
                'ID': get_image_name(self.IMG[idx]),
                'image': self.IMG[idx]
            }

        if self.transform is not None:
            data = self.transform(data)

        return data


def get_train_valid_dataset(csvroot, args):
    train_subject, train_image, _ = ReadCSV(
        os.path.join(csvroot, f'Train_{args.dataset}.csv'))
    valid_subject, valid_image, _ = ReadCSV(
        os.path.join(csvroot, f'Valid_{args.dataset}.csv'))

    TrainDataset = BPDataset(
        root=csvroot,
        SUBJECT=train_subject,
        IMG=train_image,
        transform=get_transforms(args, 'train'))
    ValidDataset = BPDataset(
        root=csvroot,
        SUBJECT=valid_subject,
        IMG=valid_image,
        transform=get_transforms(args, 'valid'))

    return TrainDataset, ValidDataset


def get_test_dataset(args):
    test_image_list = glob.glob(os.path.join('dataset', 'test', '*.tif'))
    test_image_list = sorted(test_image_list, key=lambda x: get_image_name(x))
    test_dataset = BPDataset(
        root=None,
        SUBJECT=None,
        IMG=test_image_list,
        transform=get_transforms(args, 'test'),
        training=False)

    return test_dataset


def get_transforms(args, TYPE):
    if TYPE.lower() == 'train':
        transform = Compose([
            LoadImage(keys=['image', 'label']),
            ResizeImage(keys=['image', 'label'], size=(448, 576)),
            ImageToTensor(keys=['image', 'label']),
            RandFliplr(keys=['image', 'label'], p=args.fliplr),
            RandFlipud(keys=['image', 'label'], p=args.flipud),
            RandRot90(keys=['image', 'label'], p=args.rot90),
            RandomBrightness(keys=['image'], p=args.bright),
            RandomGaussianNoise(keys=['image'], p=args.noise)
        ])

    elif TYPE.lower() == 'valid':
        transform = Compose([
            LoadImage(keys=['image', 'label']),
            ResizeImage(keys=['image', 'label'], size=(448, 576)),
            ImageToTensor(keys=['image', 'label'])
        ])

    else:
        transform = Compose([
            LoadImage(keys=['image']),
            ResizeImage(keys=['image'], size=(448, 576)),
            ImageToTensor(keys=['image'])
        ])

    return transform
