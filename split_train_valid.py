import os
import csv
from collections import defaultdict

import cv2
import numpy as np

import src.RLE as RLE
from src.utils import ReadCSV


class TrainMask(object):
    def __init__(self, subject, img, pixels):
        self.img = int(img)
        self.pixels = pixels
        self.subject = int(subject)

    def empty_label(self):
        return not len(self.pixels)

    def __lt__(self, other):
        return self.img < other.img

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
    The goal of this script is construct a csv file by
    1. inherit from "train_masks.csv"
    2. clean up false negative data (remove some rows from step 1)
    '''
    # set up the path and load the content from "train_masks.csv"
    trainPath = os.path.join('dataset', 'train')
    csvPath = os.path.join('dataset', 'train_masks.csv')
    SUBJECT, IMG, PIXELS = ReadCSV(csvPath)

    # Initialize some variables
    pos = defaultdict(lambda: [])
    neg = defaultdict(lambda: [])
    subject = defaultdict(lambda: [])

    # collect all subject and image ID, store as dict(subID->list[imgID])
    for i in range(len(IMG)):
        pix = PIXELS[i]
        imgID = IMG[i]
        subID = SUBJECT[i]
        subject[subID].append(TrainMask(subID, imgID, pix))

    # dividing into positive and negative parts (wheter label is empty or not)
    for subID, maskList in subject.items():
        for mask in maskList:
            if mask.empty_label():
                neg[subID].append(mask)
            else:
                pos[subID].append(mask)

    # remove False negative from the dataset
    for subID, maskList in neg.items():
        print(f"Processing subjectID: {subID}")

        # Read all positive images for that subject ID
        posImgs = []
        for mask in pos[subID]:
            imgpath = os.path.join(trainPath, f"{subID}_{mask.img}.tif")
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            posImgs.append(img)
        posImgs = np.array(posImgs)

        # Check whether exist identical images
        for mask in maskList:
            imgpath = os.path.join(trainPath, f"{subID}_{mask.img}.tif")
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            comp = posImgs == img

            for i, j in enumerate(comp):
                if j.all():
                    print(pos[subID][i].img, mask.img)
                    neg[subID].remove(mask)
                    break

    # dump everything to a new csv file
    for subID in subject.keys():
        subject[subID] = pos[subID] + neg[subID]
        subject[subID] = sorted(subject[subID])

    content = []
    for subID, maskList in subject.items():
        for mask in maskList:
            content.append(mask)

    WriteCSV(path=os.path.join('dataset', 'clean_masks.csv'),
             DATA=content)
