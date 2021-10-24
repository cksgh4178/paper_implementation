# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:58:53 2021

@author: USER
"""

# 모델 학습 및 평가를 위한 hyperparameter 설정

import torch
from src.RN_implementation.train import train
# =============================================================================
# 학습 관련 설정

epochs = 3000
learning_rate = 2.5e-4
batch_size = 64
weight_decay = 0
device = torch.device('cuda')
# =============================================================================

# =============================================================================
# LSTM 관련 설정

# word embedding 차원
word_emb = 32

# lstm layer 수
lstm_layers = 1

# lstm hidden state 차원
lstm_hidden_dim = 128
# =============================================================================

# =============================================================================
# RN 관련 설정

# MLP 활성화 함수 설정 (기본은 ReLu, True일 경우 tanh 사용)
mlp_tanh = False 

# MLP g 설정
g_hidden_dims = [256,256,256,256]
g_output_dim = 256

# MLP f 설정
f_hidden_dims = [256,256,32]
# =============================================================================

avg_train_accuracies, avg_train_losses, val_accuracies, val_losses = \
    train(epochs,learning_rate, batch_size, weight_decay, device, word_emb, lstm_layers,
      lstm_hidden_dim, mlp_tanh, g_hidden_dims, g_output_dim, f_hidden_dims)
    
print('Max train: ', max(avg_train_accuracies), 'Max val: ', max(val_accuracies))
