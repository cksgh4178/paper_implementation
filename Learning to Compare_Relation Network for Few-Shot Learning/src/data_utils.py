# -*- coding: utf-8 -*-
"""
Date: Wed Dec  1 20:29:46 2021

@author: Park Chan Ho
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

# data folder root
def omniglot_character_folder():
    data_dir = '../data/omniglot'
    character_folder = [os.path.join(data_dir, family, character) \
                    for family in os.listdir(data_dir) \
                        if os.path.isdir(os.path.join(data_dir, family)) \
                            for character in os.listdir(os.path.join(data_dir, family))]
    random.seed(1)
    random.shuffle(character_folder)
    
    num_train = 1200 # in paper training 1200 test 423
    meta_train_folders = character_folder[:num_train]
    meta_test_folders = character_folder[num_train:]
    
    return meta_train_folders, meta_test_folders

# Rotataion
class Rotate:
    
    def __init__(self, angle):
        
        self.angle = angle
    
    def __call__(self, x, mode = 'reflect'):
        
        x = x.rotate(self.angle)
        
        return x
# =============================================================================
# Task Definition(random select per folder(train/test))
# N-way K-shot
# N : num_classes
# K : train_num
# query size : test_num
# =============================================================================
class OmniglotTask:
    
    def __init__(self, character_folder, num_classes, train_num, test_num):
        
        self.character_folder = character_folder
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        
        # N way select
        class_folder = random.sample(character_folder, num_classes)
        labels = np.array(range(len(class_folder)))
        labels = dict(zip(class_folder, labels))
        
        # K, query sample select
        sample = dict()
        
        # K shot root
        self.train_root = list()
        
        # query root
        self.test_root = list()
        
        for c in class_folder:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            sample[c] = random.sample(temp, len(temp))
            
    
            self.train_root += sample[c][:train_num]  
            self.test_root += sample[c][train_num:train_num+test_num]
            
        self.train_label = [labels[os.path.dirname(x)] for x in self.train_root]
        self.test_label = [labels[os.path.dirname(x)] for x in self.test_root]

# =============================================================================
# Task to Dataset
# read root dir and load image
# image to tensor (transform)
# =============================================================================
class Omniglot_Dataset(Dataset):
    
    def __init__(self, task, split='train', transform=None, label_transform=None):
        
        self.task = task
        self.transform = transform
        self.label_transform = label_transform
        self.split = split
        self.roots = task.train_root if split == 'train' else task.test_root
        self.labels = task.train_label if split == 'train' else task.test_label
    
    def __len__(self):
        
        return len(self.roots)
    
    def __getitem__(self, idx):
        
        image = Image.open(self.roots[idx])
        image = image.convert('L')
        image = image.resize((28,28), resample=Image.LANCZOS) # in paper setting
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.label_transform is not None:
            label = self.label_transform(label)
            
        return image, label

# =============================================================================
# Sampler
# num_classes : N
# num_per_class : K or query size
# =============================================================================
class ClassBalancedSampler(Sampler):
    
    def __init__(self, num_per_class, num_classes, shuffle=True):
        
        self.num_per_class = num_per_class
        self.num_classes = num_classes
        self.shuffle = shuffle
        
    def __iter__(self):
        
        if self.shuffle:
            batch = [[i+j*self.num_per_class for i in torch.randperm(self.num_per_class)] \
                     for j in range(self.num_classes)]
        else:
            batch = [[i+j*self.num_per_class for i in range(self.num_per_class)] \
                     for j in range(self.num_classes)]
                
        batch = [item for sublist in batch for item in sublist]
        
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)
    
    def __len__(self):
        
        return 1

def get_data_loader(task, num_per_class=1, split='train', shuffle=True, rotation=0):
    normalize = transforms.Normalize(mean=0.92206, std=0.08426)
    dataset = Omniglot_Dataset(task, split=split, transform = \
                               transforms.Compose([Rotate(rotation), transforms.ToTensor(), normalize]))
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, shuffle)
    loader = DataLoader(dataset, batch_size = num_per_class * task.num_classes, sampler = sampler)
    
    return loader

        
        