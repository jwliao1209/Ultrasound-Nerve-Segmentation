import os
import glob
import tqdm
import argparse
import numpy as np
from PIL import Image
from src.RLE import *
from src.logger import Logger
from src.utils import *
from src.configs import *
from src.dataset import *
from src.metrics import *
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F


def set_model(args, device):
    weight_list = sorted(
        glob.glob(os.path.join('checkpoint', args.checkpoint, '*.pth')),
        key=lambda x: float(x.split('/')[-1][-10:-4]), reverse=True)

    weight_list = weight_list[:args.ensem_num]
    model_list = [get_model(args) for weight in range(len(weight_list))]
    for model, weight in zip(model_list, weight_list):
        model.load(weight)
        model.to(device)
        model.eval()

    print(weight_list)

    return model_list


def compute_sim(x, y):
    x = torch.argmax(x, dim=1)
    y = torch.argmax(y, dim=1)
    acc = compute_dice(x, y)

    return float(acc.mean().cpu().numpy())


def compute_dice_sim_matrix(preds):
    n = len(preds)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 0.5

            elif i < j:
                A[i, j] = compute_sim(preds[i], preds[j])

            else:
                continue

    dice_sim_matrix = A + A.T

    return dice_sim_matrix


def compute_first_ev(matrix):
    ew, ev = np.linalg.eig(matrix)
    max_index = np.argmax(ew)
    first_ev = ev[:, max_index]

    return first_ev


def softmax(vec):
    a = 100
    e = a**vec

    return e / np.sum(e)


def compute_coeff(vector):
    coeff = softmax(np.abs(vector))

    return coeff


def print_matrix(M):
    m, n = np.shape(M)
    for i in range(m):
        for j in range(n):
            print(round(M[i][j], 2), end='\t')

        print('\n')

    return


def compute_weighted_sum(preds, coeff):
    preds = torch.stack([c*p for (c, p) in zip(coeff, preds)])
    pred = torch.sum(preds, dim=0)
    pred = torch.argmax(pred, dim=1)

    return pred


def fliplr(inputs):
    outputs = torch.zeros_like(inputs)

    for i in range(inputs.shape[0]):
        outputs[i, :, :, :] = F.hflip(inputs[i, :, :, :])

    return outputs


def flipud(inputs):
    outputs = torch.zeros_like(inputs)

    for i in range(inputs.shape[0]):
        outputs[i, :, :, :] = F.vflip(inputs[i, :, :, :])

    return outputs


def ensemble(args, adaptive=True):
    log = Logger(print_=False)
    test_dataset = get_test_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_list = set_model(args, device)
    Postprocess = Resize((420, 580))

    with torch.no_grad():
        for batch_data in tqdm.tqdm(test_loader):
            images = batch_data['image'].to(device)
            preds = []

            for model in model_list:
                preds += [
                    model(images),
                    fliplr(model(fliplr(images))),
                    flipud(model(flipud(images))),
                    fliplr(flipud(model(flipud(fliplr(images)))))
                ]

            pred = compute_weighted_sum(preds, [1, 1, 1, 1])
            pred = Postprocess(pred)
            pred = pred[0, :, :].cpu().numpy()

            if np.sum(pred) <= 2500:
                pred = pred*0

            elif np.sum(pred) > 0 and adaptive:
                dice_sim_matrix = compute_dice_sim_matrix(preds)
                first_ev = compute_first_ev(dice_sim_matrix)
                coeff = compute_coeff(first_ev)
                pred = compute_weighted_sum(preds, coeff)
                pred = Postprocess(pred)
                pred = pred[0, :, :].cpu().numpy()

                # print_matrix(dice_sim_matrix)
                # print(coeff)

            log.add(
                img=str(int(batch_data['ID'].numpy())),
                pixels=encode(pred)
            )

        log.save(os.path.join(args.save_root, args.checkpoint, 'answer.csv'))
        print('finish!')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive', type=bool, default=True,
                        help='adaptive ensemble')
    parser.add_argument('--checkpoint', type=str,
                        default='2021-12-31-02-33-20-best',
                        help='checkpoint')
    parser.add_argument('--ensem_num', type=int, default=1,
                        help='ensemble number')
    args = parser.parse_args()

    config = load_json(os.path.join(
        'checkpoint', args.checkpoint, 'config.json'))
    args = argparse.Namespace(**vars(args), **config)
    ensemble(args)
