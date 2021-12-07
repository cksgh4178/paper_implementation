# -*- coding: utf-8 -*-
"""
Date: Wed Dec  1 10:17:32 2021

@author: Park Chan Ho
"""
# =============================================================================
# Learning to Compare : Relation Network for Few-Shot Learning
# for Few-Shot Learning(FSL)
# Not implemented for Zero-Shot Learning(ZSL)
# Dataset : Omniglot
# =============================================================================

import torch
import torch.nn as nn
import math

# Embedding module
def conv_block_pooling(in_dim, padding):
    model = nn.Sequential(
        nn.Conv2d(in_dim, 64, kernel_size=3, padding=padding),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU(),
        nn.MaxPool2d(2))
    return model

def conv_block(padding):
    model = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=padding),
        nn.BatchNorm2d(64, momentum=1, affine=True),
        nn.ReLU()
        )
    return model

class embedding_module(nn.Module):
    
    def __init__(self, in_dim):
        super(embedding_module, self).__init__()
        self.block1 = conv_block_pooling(in_dim, 0)
        self.block2 = conv_block_pooling(64, 0)
        self.block3 = conv_block(1)
        self.block4 = conv_block(1)
        
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out

# Relation Network
class relation_network(nn.Module):
    
    def __init__(self):
        super(relation_network,self).__init__()
        self.block1 = conv_block_pooling(128, 1)
        self.block2 = conv_block_pooling(64, 1)
        self.fc_1 = nn.Sequential(
            nn.Linear(64,8),
            nn.ReLU())
        self.fc_2 = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_1(out)
        out = self.fc_2(out)
        return out
    
def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())