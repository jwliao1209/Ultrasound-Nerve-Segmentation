import os
import csv
import random
import argparse
from collections import defaultdict

from src.utils import ReadCSV

class TrainMask(object):
    def __init__(self, subject, img, pixels):
        self.img = int(img)
        self.pixels = pixels
        self.subject = int(subject)

    def empty_label(self):
        return not len(self.pixels)

    def to_csv_list(self):
        return [str(self.subject), str(self.img), self.pixels]

def WriteCSV(path, DATA):
    with open(path, 'w', newline='') as foldfile:
        writer = csv.writer(foldfile)
        writer.writerow(['subject', 'img', 'pixels'])
        for data in DATA:
            writer.writerow(data.to_csv_list())

    return

if __name__ == '__main__':
    '''
    The goal of this script is to split 'dataset/clean_masks.csv' into training
    and validation dataset.

    However, there are many empty mask in our training data
    (empty:nonempty is around 3:2)

    Our goal is as follows:
        1. Split 8:2 into Train/Valid from nonempty dataset.
        2. Random sample the same amount of "8" to Train from empty dataset.
        3. The remaining is to to Validation dataset.
    Therefore, we keep 1:1 (empty:non-empty) ratio in our dataset.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-loc', '--datasetLoc', type=str, default='dataset',
                        help='dataset location, must contain clean_masks.csv')
    parser.add_argument('-tr', '--TrainRatio', type=float, default=0.8,
                        help='the ratio of Train:Valid')
    parser.add_argument('-n', '--n', type=int, default=5,
                        help='the number of times of bagging')
    args = parser.parse_args()

    # Read 'train_masks.csv'
    path = os.path.join(args.datasetLoc, 'train_masks.csv')
    SUBJECT, IMG, PIXELS = ReadCSV(path=path)

    # split it into empty and non-empty
    posID, negID = [], []
    for i, pix in enumerate(PIXELS):
        if len(pix):
            posID.append(i)
        else:
            negID.append(i)

    # split it into n folds
    for i in range(1, args.n+1):
        # shuffle index
        random.shuffle(posID)
        random.shuffle(negID)

        # split dataset
        NumTrain = int(args.TrainRatio * len(posID))
        NumValid = len(posID) - NumTrain
        TrainID = posID[:NumTrain] + negID[:NumTrain]
        ValidID = posID[NumTrain:] + negID[NumTrain:NumTrain+NumValid]

        # shuffle
        random.shuffle(TrainID)
        random.shuffle(ValidID)

        # generate csv content
        Train, Valid = [], []
        for tid in TrainID:
            Train.append(TrainMask(SUBJECT[tid], IMG[tid], PIXELS[tid]))
        for vid in ValidID:
            Valid.append(TrainMask(SUBJECT[vid], IMG[vid], PIXELS[vid]))

        WriteCSV(path=os.path.join(args.datasetLoc, f'Train_{i}.csv'),
                 DATA=Train)
        WriteCSV(path=os.path.join(args.datasetLoc, f'Valid_{i}.csv'),
                 DATA=Valid)
