import glob
import math
import numpy as np
import os
import os.path as osp
import string
import pickle
import json

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Scale(object):
  """Scale transform with list as size params"""

  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation

  def __call__(self, img):
    return img.resize((self.size[1], self.size[0]), self.interpolation)


class data_loader(object):
    '''
    load Microsoft COCO dataset
    load train/val/test dataset
    '''
    def __init__(self, root, split = 'train', max_tokens = 15, ncap_per_img = 5):
        self.root = root
        self.split = split
        self.max_tokens = max_tokens
        self.ncap_per_img = ncap_per_img
        # self.get_split_info('data/dataset_coco.json')
        
        worddict_tmp = pickle.load(open('data/wordlist.p', 'rb'))
        wordlist = [word for word in iter(worddict_tmp.keys()) if word != '</S>']
        self.wordlist = ['EOS'] + sorted(wordlist)
        self.numwords = len(wordlist)
        print('[DEBUG] #words in wordlist: %d' % (self.numwords))
        
        '''
        torchvision.transform is used to make data augmentation
        '''
        self.img_transforms = transforms.Compose([
            Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
                std = [ 0.229, 0.224, 0.225 ])
            ])
        
    def get_split_info(self, split_file):
        print('-' * 10 )