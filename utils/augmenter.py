# encoding: utf-8
import torch
import random
import numpy as np


#TODO add documentation to all functions in this file
def tensor_random_flip(tensor):
    # (b, c, t, h, w)
    if random.random() > 0.5:
        return torch.flip(tensor, dims=[4])

    return tensor


def tensor_random_crop(tensor, size):
    h, w = tensor.size(-2), tensor.size(-1)
    tw, th = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    return tensor[:, :, :, x1:x1 + th, y1:y1 + w]


def center_crop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    x1 = int(round((w - tw)) / 2.)
    y1 = int(round((h - th)) / 2.)
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]

    return img


def random_crop(batch_img, size):
    th, tw = size
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]

    return img


def horizontal_flip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])

    return batch_img
