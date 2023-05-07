# coding: utf-8
import random
import cv2
import numpy as np
import torch


def center_crop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)    
    img = batch_img[:, y1:y1+th, x1:x1+tw]
    return img


def random_crop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    img = batch_img[:,y1:y1+th,x1:x1+tw]
    return img


def horizontal_flip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:,:,::-1])
    return batch_img
