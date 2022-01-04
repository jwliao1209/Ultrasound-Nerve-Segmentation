import torch


def compute_dice(inputs, targets):
    smooth_ep = 1e-5
    batch_size = inputs.shape[0]
    inputs = inputs.reshape(batch_size, -1)
    targets = targets.reshape(batch_size, -1)
    nume = torch.sum(inputs*targets, dim=1)
    deno = torch.sum(inputs+targets, dim=1)
    dice = (2*nume + smooth_ep) / (deno + smooth_ep)

    return dice
