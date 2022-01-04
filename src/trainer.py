import os
import glob
import tqdm
import torch
import monai
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from src.configs import *
from src.metrics import compute_dice
from src.logger import Logger
from src.utils import *


def compute_acc(x, y):
    x = torch.argmax(x, dim=1)
    acc = compute_dice(x, y)

    return acc.mean()


def save_model(root, model, epoch, acc, weight_num):
    filename = f'epoch={str(epoch):0>4}-acc={acc:.5f}.pth'
    save_path = os.path.join(root, filename)
    model.save(save_path)
    weight_list = sorted(
        glob.glob(os.path.join(root, '*.pth')),
        key=lambda x: float(x.split('/')[-1][-10:-4]), reverse=True)

    if len(weight_list) > weight_num:
        os.remove(weight_list[-1])

    return None


def training_step(model, train_loader, criterion, optimizer, device):
    loss_list, acc_list = [], []
    model.train()
    for batch_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()

        images = batch_data['image'].to(device)
        labels = batch_data['label'].to(device)
        preds = model(images)
        losses = criterion(preds, labels)
        acc = compute_acc(preds, labels)

        losses.backward()
        optimizer.step()

        loss_list.append(losses.item())
        acc_list.append(acc.item())

    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(acc_list)

    return mean_loss, mean_acc


def validation_step(model, valid_loader, criterion, device):
    loss_list, acc_list = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm.tqdm(valid_loader):
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)
            preds = model(images)
            losses = criterion(preds, labels)
            acc = compute_acc(preds, labels)
            loss_list.append(losses.item())
            acc_list.append(acc.item())

    mean_loss = np.mean(loss_list)
    mean_acc = np.mean(acc_list)

    return mean_loss, mean_acc


def trainer(args, model, train_dataset, valid_dataset):
    cur_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    save_root = os.path.join(args.save_root, cur_time)
    os.makedirs(save_root, exist_ok=True)
    save_json(args, os.path.join(save_root, 'config.json'))

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    criterion = get_criterion(args)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4)

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4)

    model.to(device)
    log = Logger()
    save_model(save_root, model, 0, 0, args.weight_num)

    for ep in range(1, args.epoch+1):
        train_loss, train_acc = training_step(
            model, train_loader, criterion, optimizer, device)
        log.add(
            epoch=ep,
            type='train',
            loss=train_loss,
            dice=train_acc,
            lr=scheduler.get_last_lr()[0]
        )

        valid_loss, valid_acc = validation_step(
            model, valid_loader, criterion, device)
        log.add(
            epoch=ep,
            type='valid',
            loss=valid_loss,
            dice=valid_acc,
            lr=scheduler.get_last_lr()[0]
        )

        scheduler.step()
        save_model(save_root, model, ep, valid_acc, args.weight_num+1)
        log.save(os.path.join(save_root, 'record.csv'))

    return None
