import numpy as np


def encode(img):
    '''
    Input:
        img (ndarray, uint8): a mask of shape (height, weight)
    Output:
        code (str): RLE encoding with the format in this competition
    '''
    height, width = img.shape
    # The RLE ID in competitions starts from 1
    indices = np.where(img.flatten(order='F'))[0] + 1
    prev, code = -1, []
    for ind in indices:
        if (ind > prev + 1):
            code.append(ind)
            code.append(0)
        code[-1] += 1
        prev = ind
    code = map(str, code)
    code = ' '.join(code)
    return code


def decode(code, width=580, height=420):
    '''
    Inputs:
        code (str): RLE encoding with the format in this competition
                    e.g: '168153 9 168570 15 168984 22'
        width (int)
        height (int)
    Outputs:
        img (ndarray[uint8]): a mask
    '''
    # The rule in competition starts from 1: 1->(1,1), 2->(2,1) ...
    # However, our index starts from zero  : 1->(0,0), 2->(1,0) ...
    # pad one zero to the 1d array and cancel the first element.
    img = np.zeros(width * height + 1, dtype=np.uint8)

    if len(code) > 1:
        code = list(map(lambda x: int(x), code.split(' ')))
        start_pixs, run_lens = code[0::2], code[1::2]
        for p, l in zip(start_pixs, run_lens):
            img[p:p+l] = 1

    img = img[1:].reshape(height, width, order='F')

    return img
