import os
import csv
import json


def save_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(vars(obj), fp, indent=4)

    return None


def load_json(path):
    with open(path, 'r') as fp:
        obj = json.load(fp)

    return obj


def ReadCSV(path=os.path.join('dataset', 'train_masks.csv')):
    '''
    Read the content from train_masks.csv
    Then return some information for '{subject}_{img}_mask.tif'
    Input:
        path (str): input path
    Output:
        subject (list[str]): list of {subject}
        img (list[str]): list of {img}
        pixels (list[str]): list of RLE codes from {subject}_{img}_mask.tif
    '''
    subject, img, pixels = [], [], []
    with open(path, 'r') as fp:
        csvfile = csv.reader(fp, delimiter=',')
        for i, row in enumerate(csvfile):
            # row = ['{subject}', '{img}', 'RLE'] here

            if i == 0:
                continue
            img.append(row[1])
            pixels.append(row[2])
            subject.append(row[0])

    return subject, img, pixels
