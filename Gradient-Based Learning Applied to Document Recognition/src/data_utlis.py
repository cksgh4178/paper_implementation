# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:34:11 2021

@author: Park Chan Ho
"""

from torchvision import datasets, transforms

# MNIST 데이터 읽어오기
train_data = datasets.MNIST(root = '../data', train=True, download=True)
