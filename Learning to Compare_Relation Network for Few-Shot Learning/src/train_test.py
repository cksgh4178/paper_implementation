# -*- coding: utf-8 -*-
"""
Date: Mon Dec  6 14:07:37 2021

@author: Park Chan Ho
"""

import torch
import random
import src.data_utils as util

def make_data_loaders(root, num_classes, train_num, test_num):
    
    degree = random.choice([0,90,180,270])
    task = util.OmniglotTask(root, num_classes, train_num, test_num)
    sample_loader = util.get_data_loader(task, num_per_class=train_num, 
                                         split = 'train', shuffle=False, rotation=degree)
    query_loader = util.get_data_loader(task, num_per_class=test_num, 
                                        split='test', shuffle=True, rotation=degree)
    
    return sample_loader, query_loader

def cal_correct(pred, true):
    
    _, pred_label = torch.max(pred.data, 1)
    reward = [1 if pred_label[i] == true[i] else 0 for i in range(true.size(0))]
    
    return reward

def train(feature_module, relation_module, feature_optim, relation_optim, criterion, 
          root, NUM_CLASS, SAMPLE_PER_CLASS, QUERY_SIZE, device, wandb):
    
    print('training...')
    
    feature_module.train()
    relation_module.train()
    
    # data loaders
    sample_loader, query_loader = make_data_loaders(root, NUM_CLASS, SAMPLE_PER_CLASS, QUERY_SIZE)
    
    # get next data
    samples, sample_labels = sample_loader.__iter__().next()
    queries, query_labels = query_loader.__iter__().next()
    
    # calculate feature
    sample_feature = feature_module(samples.to(device)) # (num_class*sample, 64, 5, 5)
    sample_feature = sample_feature.view(NUM_CLASS, SAMPLE_PER_CLASS, 64, 5, 5) # (num_class, sample, 64, 5, 5)
    sample_feature = torch.sum(sample_feature, 1).squeeze(1)
    query_feature = feature_module(queries.to(device))
    
    # concat relation pair
    sample_feature_ext = sample_feature.unsqueeze(0).repeat(NUM_CLASS*QUERY_SIZE, 1, 1, 1, 1)
    query_feature_ext = query_feature.unsqueeze(0).repeat(NUM_CLASS, 1, 1, 1, 1)
    query_feature_ext = query_feature_ext.transpose(0, 1)
    relation_pair = torch.cat((sample_feature_ext, query_feature_ext), 2).view(-1, 128, 5, 5)
    
    #  calculate relation
    relations = relation_module(relation_pair).view(-1, NUM_CLASS)
    one_hot_label = torch.zeros(QUERY_SIZE*NUM_CLASS, NUM_CLASS).scatter(1, query_labels.view(-1,1).long(), 1).to(device)
    
    # calculate loss
    loss = criterion(relations, one_hot_label)
    
    # calculate correct
    reward = cal_correct(relations, query_labels)
    train_acc = sum(reward) / 1.0/NUM_CLASS/QUERY_SIZE
    
    wandb.log({'train_acc':train_acc,
               'train_loss':loss.item()})
    
    # weight update
    feature_optim.zero_grad()
    relation_optim.zero_grad()
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm(feature_module.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm(relation_module.parameters(), 0.5)
    
    feature_optim.step()
    relation_optim.step()

def test(TEST_EPISODE, feature_module, relation_module, root, NUM_CLASS, SAMPLE_PER_CLASS, 
         device, wandb):
    
    print('testing...')
    
    feature_module.eval()
    relation_module.eval()
    total_reward = 0.
    
    for i in range(TEST_EPISODE):
        
        # In test, train_num = test_num
        sample_loader, query_loader = make_data_loaders(root, NUM_CLASS, SAMPLE_PER_CLASS, SAMPLE_PER_CLASS)
        
        samples, sample_labels = sample_loader.__iter__().next()
        queries, query_labels = query_loader.__iter__().next()
        
        sample_feature = feature_module(samples.to(device)) # (num_class*sample, 64, 5, 5)
        sample_feature = sample_feature.view(NUM_CLASS, SAMPLE_PER_CLASS, 64, 5, 5) # (num_class, sample, 64, 5, 5)
        sample_feature = torch.sum(sample_feature, 1).squeeze(1)
        query_feature = feature_module(queries.to(device))
        
        sample_feature_ext = sample_feature.unsqueeze(0).repeat(NUM_CLASS*SAMPLE_PER_CLASS, 1, 1, 1, 1)
        query_feature_ext = query_feature.unsqueeze(0).repeat(NUM_CLASS, 1, 1, 1, 1)
        query_feature_ext = query_feature_ext.transpose(0, 1)
        relation_pair = torch.cat((sample_feature_ext, query_feature_ext), 2).view(-1, 128, 5, 5)
        
        relations = relation_module(relation_pair).view(-1, NUM_CLASS)
        
        reward = cal_correct(relations, query_labels)
        total_reward += sum(reward)
    
    test_acc = total_reward/1.0/NUM_CLASS/SAMPLE_PER_CLASS/TEST_EPISODE
    
    wandb.log({'test_acc':test_acc})