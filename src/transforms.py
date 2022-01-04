import random
import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor
import torchvision.transforms.functional as F


class LoadImage():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Image.open(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class ResizeImage():
    def __init__(self, keys, size=(448, 576)):
        self.keys = keys
        self.size = size

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Resize(size=self.size)(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class RandFliplr():
    def __init__(self, keys, p=0.3):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            for key in self.keys:
                if key in data:
                    data[key] = F.hflip(data[key])

                else:
                    raise KeyError(f'{key} is not a key of {data}')

        return data


class RandFlipud():
    def __init__(self, keys, p=0.1):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            for key in self.keys:
                if key in data:
                    data[key] = F.vflip(data[key])

                else:
                    raise KeyError(f'{key} is not a key of {data}')

        return data


class RandRot90():
    def __init__(self, keys, p=0.1):
        self.keys = keys
        self.p = p

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            for key in self.keys:
                if key in data:
                    data[key] = torch.rot90(data[key], k=1, dims=[1, 2])

                else:
                    raise KeyError('f{key} is not a key of {data}')
        return data


class RandomBrightness():
    def __init__(self, keys, p=0.1, factor=0.1):
        self.keys = keys
        self.p = p
        self.factor = factor

    def __call__(self, data):
        if random.uniform(0, 1) <= self.p:
            factor = random.uniform(1-self.factor, 1+self.factor)
            for key in self.keys:
                if key in data:
                    data[key] = F.adjust_brightness(data[key], factor)
                    data[key] = torch.clip(data[key], min=0, max=1)
                else:
                    raise KeyError('f{key} is not a key of {data}')
        return data


class RandomGaussianNoise():
    def __init__(self, keys, p=0.1, sig=0.01):
        self.keys = keys
        self.sig = sig
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            for key in self.keys:
                if key in data:
                    data[key] += self.sig * torch.randn(data[key].shape)

                else:
                    raise KeyError('f{key} is not a key of {data}')
        return data


class ImageToTensor():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = ToTensor()(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data
