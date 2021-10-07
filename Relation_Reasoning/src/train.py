# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:53:10 2021

@author: USER
"""


import torch
from torch.utils.data import DataLoader
from src.RN_implementation.models import RN
from src.RN_implementation.utils import read_dateset, split_train_test, vectorize_stories, bAbidataset, batchify, get_answer
import wandb

wandb.init(project="research", entity="cksgh4178")
wandb.run.name = 'batch_64/2.5e-4/3000ep/128lstm'
# =============================================================================
# 네트워크 가중치 초기화 함수\
def init_weights(m):
    # if m.dim() > 1:
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
# =============================================================================

# =============================================================================
# launch(main)

def train(epochs, learninig_rate, batch_size, weight_decay, device, word_emb, lstm_layers, 
         lstm_hidden_dim, mlp_tanh, g_hidden_dims, g_output_dim, f_hidden_dims):
    

    # 데이터 준비
    # 데이터 읽어오기(학습 및 검증을 한번에)
    # 답이 2개인 문제 때문에 8,19번 빼고 진행
    stories,dictionaries,labels = read_dateset('train')
    
    # 사전크기 정의  - 워드 임베딩 시 사용
    dictionary_size = len(dictionaries)
    print('Dictionary size: ' , dictionary_size)
    
    # 학습, 검증 데이터 분리(전체 story, task별 라벨, test 비율)
    train_stories, validation_stories = split_train_test(stories, labels, 0.2)
    
    # label 제거
    train_stories = train_stories[['question', 'answer', 'sup_facts']]
    validation_stories = validation_stories[['question', 'answer', 'sup_facts']]
       
    # dictionary 이용한 vector로 변환된 tensor
    train_stories = vectorize_stories(train_stories, dictionaries, device)
    validation_stories = vectorize_stories(validation_stories, dictionaries, device)
    
    # Dataset 정의
    train_dataset = bAbidataset(train_stories)
    
    # 네트워크 구성
    rn = RN(dictionary_size, word_emb, lstm_hidden_dim, g_hidden_dims, g_output_dim, 
            f_hidden_dims, dictionary_size, mlp_tanh,batch_size, device).to(device)
    
    rn.apply(init_weights)
        
    # 최적화 및 손실함수 정의
    optimizer = torch.optim.Adam(rn.parameters(), lr = learninig_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    # 학습 정확도 및 손실(배치의 평균)
    avg_train_accuracies = []
    avg_train_losses = []
    
    # 검증 정확도 및 손실
    val_accuracies = []
    val_losses = []
    
    # 학습 시작
    print('\n--training start--\n')
    for epoch in range(1, epochs + 1):
        
        # 배치별 학습 정확도 계산 리스트
        train_accuracies = []
        train_losses = []
        
        # 배치 구성해서 데이터 받아오기
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, collate_fn=batchify, drop_last=True)
        
        rn.train()
        
        print('epoch ', epoch, ' training start')
        # 배치 별 데이터 받기
        for batch_id, (question_batch, answer_batch, facts_batch) in enumerate(train_loader):
            
            if (batch_id) % 100 == 0:
                print('Train Batch: ', batch_id, ' / ', len(train_loader), ' -epoch: ', epoch)
            
            question_batch, answer_batch, facts_batch = \
            question_batch.to(device), answer_batch.to(device), facts_batch.to(device)
                            
            # 결과 받아오기
            pred = rn(question_batch, facts_batch)
            
            rn.zero_grad()
            
            # loss 계산
            loss = criterion(pred, answer_batch.squeeze())
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 학습 진행상태 관리
            correct = get_answer(pred, answer_batch)
            train_accuracies.append(correct)
            train_losses.append(loss.item())
             
        # 배치의 평균 정확도, 손실 계산
        avg_train_losses.append(sum(train_losses)/len(train_losses))
        avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))
        
        # 검증 정확도, 손실 계산
        val_accuracy, val_loss = test(rn, validation_stories, criterion, device, batch_size)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        
        print()
        print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
        print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
        print('epoch ', epoch, ' training end\n')
        
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_losses[-1],
            'train_accuracy': avg_train_accuracies[-1],
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
            })
    print('\n--training end--')
    return avg_train_accuracies, avg_train_losses, val_accuracies, val_losses
# =============================================================================

# =============================================================================
# 평가 및 검증 함수
# train과 거의 유사

def test(rn, valid_stories, criterion, device, batch_size):
    
    with torch.no_grad():      
        valid_loss = 0.
        valid_accuracy = 0.
        
        rn.eval()
        
        valid_dataset = bAbidataset(valid_stories)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, 
                                  collate_fn=batchify, drop_last=True)
        
        for batch_id, (question_batch, answer_batch, facts_batch) in enumerate(valid_loader):

            question_batch, answer_batch, facts_batch = \
            question_batch.to(device), answer_batch.to(device), facts_batch.to(device)
                        
            pred = rn(question_batch, facts_batch)
            
            loss = criterion(pred, answer_batch.squeeze())
            
            correct = get_answer(pred, answer_batch)
            
            valid_accuracy += correct
            valid_loss += loss.item()
            
    return valid_accuracy / float(len(valid_loader)) , valid_loss/ float(len(valid_loader))
# =============================================================================
