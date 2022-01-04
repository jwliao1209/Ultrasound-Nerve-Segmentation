import random
import numpy as np
import torch

import argparse
from src.configs import *
from src.trainer import trainer
from src.dataset import get_train_valid_dataset


def train(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, valid_dataset = get_train_valid_dataset(
        csvroot='dataset', args=args)
    model = get_model(args)
    trainer(args, model, train_dataset, valid_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # set dataset and model
    parser.add_argument('--dataset', type=int, default=2,
                        help='which dataset will be used.')
    parser.add_argument('--model', type=str, default='smp_unet',
                        help='model')
    parser.add_argument('--pretrain', type=str, default='efficientnet-b1',
                        help='pretrain weight')
    parser.add_argument('--activation', type=str, default='ReLU',
                        help='decoder activation function name')

    # set optimization
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('-ep', '--epoch', type=int, default=100,
                        help='epoch')
    parser.add_argument('--loss', type=str, default='DFL',
                        help='loss function')
    parser.add_argument('--optim', type=str, default='adamw',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--scheduler', type=str, default='cos',
                        help='learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=1,
                        help='learning rate decay step size')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='learning rate decay factor')

    # set others
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--save_root', type=str, default='checkpoint',
                        help='save path')
    parser.add_argument('--weight_num', type=int, default=10,
                        help='save weight number')

    # augmentation
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Probabiliy that perform flip lr')
    parser.add_argument('--flipud', type=float, default=0.25,
                        help='Probabiliy that perform flip ud')
    parser.add_argument('--rot90', type=float, default=0,
                        help='Probabiliy that perform rotate 90 degrees')
    parser.add_argument('--bright', type=float, default=0.1,
                        help='Probabiliy that adjust brightness')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Probabiliy that add gauss noise')

    args = parser.parse_args()
    train(args)
