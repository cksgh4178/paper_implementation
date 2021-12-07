# -*- coding: utf-8 -*-
"""
Date: Thu Dec  2 13:27:21 2021

@author: Park Chan Ho
"""

import os
import sys
import wandb
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

base_dir = os.path.dirname(os.getcwd())
sys.path.append(base_dir)

from src.model import embedding_module, relation_network, init_weight
import src.data_utils as util
from src.train_test import train, test

# wandb init
wandb.init(project='Learning_to_compare', entity='cksgh4178')

# parser settings
parser = argparse.ArgumentParser(description='Few-Shot Learning')
parser.add_argument('-n', '--num_class', type=int, default=5)
parser.add_argument('-k', '--sample_per_class', type=int, default=5)
parser.add_argument('-q', '--query_size', type=int, default=15)
parser.add_argument('-e', '--episode', type=int, default=10000)
parser.add_argument('-te', '--test_episode', type=int, default=1000)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-s', '--step_size', type=int, default=500)
args = parser.parse_args()

# Config update
wandb.config.update(args)

# hyperparameter settings
NUM_CLASS = args.num_class
SAMPLE_PER_CLASS = args.sample_per_class
QUERY_SIZE = args.query_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
STEP_SIZE = args.step_size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    
    # data folder selection
    print('init data folders')
    meta_train_folder, meta_test_folder = util.omniglot_character_folder()
    
    # network init
    print('Network module init')
    feature_module = embedding_module(1)
    relation_module = relation_network()
    
    # weight init
    feature_module.apply(init_weight)
    relation_module.apply(init_weight)
    
    # to device
    feature_module.to(device)
    relation_module.to(device)
    
    # loss function and optimizer
    criterion = nn.MSELoss()
    feature_optim = torch.optim.Adam(feature_module.parameters(), lr = LEARNING_RATE)
    relation_optim = torch.optim.Adam(relation_module.parameters(), lr = LEARNING_RATE)
    feature_scheduler = StepLR(feature_optim, step_size=STEP_SIZE, gamma=0.5)
    relation_scheduler = StepLR(relation_optim, step_size=STEP_SIZE, gamma=0.5)
    
    for episode in range(EPISODE):
        feature_scheduler.step(episode)
        relation_scheduler.step(episode)
        
        train(feature_module, relation_module, feature_optim, relation_optim, criterion,
              meta_train_folder, NUM_CLASS, SAMPLE_PER_CLASS, QUERY_SIZE, device, wandb)
        
        if (episode+1) % 1000 == 0:
            test(TEST_EPISODE, feature_module, relation_module, meta_test_folder,
                 NUM_CLASS, SAMPLE_PER_CLASS, device, wandb)
    
if __name__ == '__main__':
    main()