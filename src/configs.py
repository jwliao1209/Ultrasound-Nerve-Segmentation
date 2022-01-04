import math
import torch
import monai
from src.model import *


def get_model(args):
    Model = {
        'smp_unet': SMPUNet,
        'smp_unetpp': SMPUNetPlusPlus,
        'deeplabv3pp': DeepLabV3Plus
    }
    model = Model[args.model](args)

    return model


def get_optimizer(args, model):
    Optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
    }
    optimizer = Optimizer[args.optim](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    return optimizer


def get_scheduler(args, optimizer):
    Scheduler = {
        'step': torch.optim.lr_scheduler.StepLR(
              optimizer=optimizer,
              step_size=args.step_size,
              gamma=args.gamma
        ),
        'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epoch
        )
    }
    scheduler = Scheduler[args.scheduler]

    return scheduler


def get_criterion(args):
    Losses = {
        'DL':   monai.losses.DiceLoss,
        'GDL':  monai.losses.GeneralizedDiceLoss,
        'DCEL': monai.losses.DiceCELoss,
        'DFL':  monai.losses.DiceFocalLoss
    }
    criterion = Losses[args.loss](
        to_onehot_y=True,
        softmax=True)

    return criterion
