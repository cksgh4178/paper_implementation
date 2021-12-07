# Learning to Compare : Relation Network for Few-Shot Learning
<br>

* Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H. S., & Hospedales, T. M. (2017). Learning to compare: Relation network for few-shot learning. 

* Relation Network를 Few-Shot Learning에 적용한 논문

* **Meta learning** 방법의 일종으로 그 중 **Metric based learning** 에 해당
<br>

#### Few-Shot Learning (FSL) 이란?

<img src="https://www.borealisai.com/media/filer_public_thumbnails/filer_public/50/6a/506a0057-93f9-4d7a-9f91-14c4f0c8339f/t2_figure1.png__3000x1372_q85_subject_location-1500%2C686_subsampling-2.png" width="900" height="500">

* 보통 N-way K-shot 문제로 불리며 N개의 클래스에서 각각 K개만큼의 sample을 보고 query set의 class 를 예측하는 문제
* 논문에서는 **5-way 1-shot (19 query)**, **5-way 5-shot(15 query)**, **20-way 1shot (10 query)**, **20-way 5-shot(5 query)** 를 실험
* Zero-Shot Learning (ZSL)에 대한 실험도 진행하였으나 이 부분은 구현하지 않음
* ZSL 의 경우 DNN을 통해 sample image에 대한 설명을 feature로 입력
<br>

#### 모델 구조

<center><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/bfe284e4338e62f0a61bb33398353efd687f206f/4-Figure1-1.png" width = "650" height="350"></center>

* embedding module로 sample 과 query의 특징을 추출	

* relation pair를 만들어 relation module에서 relation score 계산

* relation score로 계산하기 때문에 일반적인 cross entropy loss 가 아니라 MSE loss 사용

* relation score 
  ![figure](https://latex.codecogs.com/png.latex?%5Cinline%20%5Clarge%20r_%7Bi%2C%20j%7D%20%3D%20g_%7B%5Cphi%7D%28%5Cmathbb%7BC%7D%28f_%7B%5Cvarphi%7D%28x_i%29%2C%20f_%7B%5Cvarphi%7D%28x_j%29%29%29%2C%20%5Cqquad%20i%20%3D%201%2C%202%2C%20%5Cdots%2C%20%5Cmathbb%7BC%7D)

* objective function
![그림](https://latex.codecogs.com/png.latex?%5Cinline%20%5Clarge%20%5Cvarphi%2C%20%5Cphi%20%5Cleftarrow%20%5Cunderset%7B%5Cvarphi%2C%20%5Cphi%7D%7Bargmin%7D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20%28r_%7Bi%2Cj%7D-1%28y_i%20%3D%3D%20y_j%29%29%5E2)

* 모델 세부 구조

  ![그림](https://d3i71xaburhd42.cloudfront.net/bfe284e4338e62f0a61bb33398353efd687f206f/5-Figure2-1.png)
<br>

* 모델 성능 (논문)

![그림](https://d3i71xaburhd42.cloudfront.net/bfe284e4338e62f0a61bb33398353efd687f206f/6-Table1-1.png)
<br>

#### Metric based approach
* Meta learning에서 metric based approach란 각 Task의 특징을 잘 구분할 수 있는 metric을 획득하는 것
* 새로운 Task가 들어오면 기존 Task와의 거리 (metric) 을 계산
* 일반적인 metric based 방법론은 고정된 metric (euclidean 등) 을 사용하기 때문에 *non-parametic* 한 방법
* Relation Network를 사용한 방법은 **학습할 수 있는** metric (relation score)를 사용
<br>

#### 구현 (논문 저자 공식 구현 참고)

##### model.py

```python
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
```

* 64채널의 3X3 conv layer가 반복되므로 함수로 구현
* conv의 padding 과 batchnorm의 momentum, affine 은 공식 구현 참고 (padding 필수적으로 필요)
* Relation Network의 첫 conv layer는 두 개의 feature가 concat 되어 들어오므로 64 * 2 차원
* Relation Network의 fully connected layer 들어오기 전에 view를 통해 차원 변경 필수

##### data_utlis.py

```python
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
```

* data의 root만 관리하고 있다가 접근할 때 읽어오는 방식으로 메모리 절약 (Dataset 부분 참조)
* Sampler 를 통해 class 별 sample_size, quert_size 에 맞도록 데이터를 선택할 수 있도록 구현
* 총 1625개의 character 중 1200개를 train으로 사용하고 425개를  test에 사용

##### 실험결과

* 5-way 5-shot (15 query)

![5 5](https://user-images.githubusercontent.com/23060537/144982797-6717c826-0ebe-4755-bf43-52e09648cd63.png)

* 5-way 1-shot (19 query)

![5 1](https://user-images.githubusercontent.com/23060537/144982865-c79498ef-42be-470f-9724-287da837ae45.png)

* 20-way 5-shot (5 query)

![20 5](https://user-images.githubusercontent.com/23060537/144983183-3b8f9a8c-25e7-44d4-a67c-432f9eec3659.png)

* 20-way 1-shot (10 query)

![20 1](https://user-images.githubusercontent.com/23060537/144982923-8df195f8-1bd0-42fd-8a85-6259b5b6519a.png)