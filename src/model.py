import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import monai
from monai.networks.layers import Norm


def replace_decoder_activation(model, act_name):
    if act_name.upper() == 'RELU':
        activation = nn.ReLU

    elif act_name.upper() == 'LRELU':
        activation = nn.LeakyReLU

    elif act_name.upper() == 'SILU':
        activation = nn.SiLU

    elif act_name.upper() == 'MISH':
        activation = nn.Mish

    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, activation(inplace=True))
        else:
            replace_decoder_activation(child, act_name)
    return


class BaseModule(nn.Module):
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)

        return None

    def save(self, path):
        torch.save(self.model.state_dict(), path)

        return None

    def forward(self, x):
        output = self.model(x)

        return output


class SMPUNet(BaseModule):
    def __init__(self, args):
        super(SMPUNet, self).__init__()
        self.args = args
        self.model = smp.Unet(
            encoder_name=args.pretrain,
            encoder_weights='imagenet',
            in_channels=1,
            classes=2,
            activation=None
        )
        if 'activation' in args:
            replace_decoder_activation(self.model, args.activation)


class SMPUNetPlusPlus(BaseModule):
    def __init__(self, args):
        super(SMPUNetPlusPlus, self).__init__()
        self.args = args
        self.model = smp.UnetPlusPlus(
            encoder_name=args.pretrain,
            encoder_weights='imagenet',
            in_channels=1,
            classes=2,
            activation=None
        )


class DeepLabV3Plus(BaseModule):
    def __init__(self, args):
        super(DeepLabV3Plus, self).__init__()
        self.args = args
        self.model = smp.DeepLabV3Plus(
            encoder_name=args.pretrain,
            encoder_weights='imagenet',
            in_channels=1,
            classes=2,
            activation=None
        )
