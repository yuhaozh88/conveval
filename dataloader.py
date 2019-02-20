from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import os
import os.path as osp
import string
import pickle
import json

import random
import copy

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Scale(object):
    '''
    class Scale is used to transform images with size list
    example:
    s = Scale([24, 24])
    img = s(img)
    '''
    def __init__(self, size, interpolation = Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        return img.resize((self.size[1], self.size[0]), self.interpolation)

class data_loader(Dataset):
    '''
    load train/val/test splits of coco dataset
    '''

    def __init__(self, root, split = 'train', max_tokens = 15, ncap_per_img = 5):
        self. max_tokens = max_tokens
        
    
