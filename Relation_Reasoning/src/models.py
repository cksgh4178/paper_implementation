# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:02:33 2021

@author: USER
"""

import torch
import torch.nn as nn

# =============================================================================
# MLP class

class MLP(nn.Module):
    
    # 초기화
    # 입력층 차원, 은닉층 차원들, 출력층 차원, tanh 여부를 입력으로 받음
    # 과적합 때문에 dropout 추가 
    
    def __init__(self, input_dim, hidden_dims, output_dims, mlp_tanh, dropout=False): 
        super(MLP, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        self.dropouts = nn.ModuleList( [nn.Dropout(p=0.5) for _ in range(len(self.hidden_dims))])
        self.dropout = dropout
        
        for i in range(1, len(hidden_dims)):
            self.linears.append(nn.Linear(hidden_dims[i-1],hidden_dims[i]))        
        self.linears.append(nn.Linear(hidden_dims[-1], output_dims))
        
        if mlp_tanh:
            self.activation = torch.tanh
        else:
            self.activation = torch.relu

    def forward(self, x):        
        for i in range(len(self.hidden_dims)):          
            x = self.linears[i](x)
            x = self.activation(x)
            
            if self.dropout:
                x = self.dropouts[i](x)
        
        x = self.linears[-1](x)
        out = self.activation(x)
        
        return out 
# =============================================================================

# =============================================================================
# RN

class RN(nn.Module):
    
    # 초기화
    # LSTM에 넣을 dictionarydm 크기 및 embedding의 차원, LSTM hidden state의 차원
    # MLP g, f의 입출력, 은닉층 차원 및 tanh 활성화 함수 여부, batch size
    def __init__(self, dict_size, word_emb, lstm_hidden_dim, g_hidden_dims, g_output_dim, f_hidden_dims, f_output_dim, mlp_tanh, batch_size, device):
        super(RN, self).__init__()
        
        # embedding (사전 크기 + 1 (패딩때문에), embedding 차원)
        self.embedding = nn.Embedding(dict_size + 1, word_emb)
        
        # LSTM q 설정 (질문 처리)
        self.lstm_q = nn.LSTM(word_emb, lstm_hidden_dim, batch_first = True)
        
        # LSTM f 설정 (sup_fact들 처리)
        self.lstm_f = nn.LSTM(word_emb, lstm_hidden_dim, batch_first = True)
                                
        # MLP g 설정
        # g의 input은 object1, object2, query + 2*sep
        self.g = MLP(3*(lstm_hidden_dim)+40, g_hidden_dims, g_output_dim, mlp_tanh)
        
        # MLP f 설정 (f의 입력 차원은 g의 출력차원과 같다)
        self.f = MLP(g_output_dim, f_hidden_dims, f_output_dim, mlp_tanh, dropout=True)
        
        # batch size 설정 (tensor 조작 시 사용)
        self.batch_size = batch_size
        
        # g_output_dim
        self.g_output_dim = g_output_dim
        
        # lstm hiden state
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # device
        self.device = device
        
    def forward(self, question_batch, facts_batch):
        
        # 임베딩
        # question embedding 결과: 배치 크기 * 문장 길이 * 임베딩 차원
        question_emb = self.embedding(question_batch)
        facts_emb = self.embedding(facts_batch)
        
        # LSTM을 통해 object detection
        # question hidden state 가져오기(제일 마지막 hidden state 가져오기)
        question_hidden = self.lstm_q(question_emb)[1][0].squeeze()
        
        # fact hidden state 가져오기(제일 마지막 hidden state 가져오기)
        facts_hidden = self.lstm_f(facts_emb)[1][0].squeeze()
        
        # tensor view를 통해 배치 크기 * fact 개수 * lstm hidden state 차원
        facts_hidden = facts_hidden.view(self.batch_size, -1, self.lstm_hidden_dim)
        
        # 문장갯수
        n_facts = facts_hidden.size(1)
        
        # 구분자 만들기 (20개 고정)
        sep = torch.eye(20)[:n_facts].to(self.device)
        sep = sep.repeat(self.batch_size, 1, 1)
        facts_hidden = torch.cat((facts_hidden, sep), dim = 2)
        
        # object oi- 반복하는 이유는 문장쌍을 만들어주기 위해서
        oi = facts_hidden.repeat(1, n_facts, 1)
        
        # object oj 
        oj = facts_hidden.unsqueeze(2).repeat(1,1,n_facts,1)
        oj = oj.view(self.batch_size,-1,self.lstm_hidden_dim + 20)
        
        # question cat하기 위한 준비
        question_hidden = question_hidden.unsqueeze(1).repeat(1,n_facts*n_facts,1)
        
        # object pair 만들기 (oi, oj, q)
        pair_cat = torch.cat((oi,oj,question_hidden), dim=2)
        
        # Relation 계산
        relation = self.g(pair_cat.view(-1,pair_cat.size(2)))
        relation = relation.view(self.batch_size, -1, self.g_output_dim)
        
        # RN 결과 계산 (관계를 결과로 변환)
        # sum을 해서 여러 문장의 결과를 하나로 합치기
        relation_emb = torch.sum(relation, dim = 1)
        
        out = self.f(relation_emb)
        
        return out
# =============================================================================