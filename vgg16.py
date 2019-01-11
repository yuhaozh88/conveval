import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

pretrained_model = models.vgg16(pretrained = True)

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.features_nopool = nn.Sequential(*list(pretrained_model.features.children())[:-1])
        self.features_pool = list(pretrained_model.features.children())[-1]
        self.classifier = nn.Sequential(*list(pretrained_model.classifier.children())[:-1])
    
    def forward(self, x):
        '''
        param x is a tensor represents input or feature map
        '''
        x = self.features_nopool(x)
        x_pool = self.features_pool(x)
        x_feat = x_pool.view(x_pool.size(0), -1)
        y = self.classifier(x_feat)
        return x_pool, y
