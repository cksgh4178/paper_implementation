# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:53:18 2021

@author: USER
"""


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

dictionary_size = 0
# =============================================================================
# train, test data를 읽어오는 함수
# train, test를 입력받음

def read_dateset(data_set):
    
    # 데이터 위치
    base_dir = os.path.split(os.path.split(os.getcwd())[0])[0]
    data_dir = os.path.join(base_dir, 'data')
    
    # 어떤 데이터를 읽어올 것인가(언어는 english 고정, test, train만 선택)
    d_dir = os.path.join(data_dir,'en',data_set) 
    
    # 일단 전체 파일을 다 읽는 함수로 구현(추후 task의 갯수만큼 읽어오도록 수정)
    file_list = pd.Series(os.listdir(d_dir))
    
    # return parameters
    label = 0
    dictionary = pd.Series()
    stories = pd.DataFrame()
    
    # 파일별로 읽어오는 반복문
    for name in file_list:
        file = pd.read_csv(os.path.join(d_dir,name), sep = '\t', names = ['text','answer','sup'])     
        file[['ID','text']] = file['text'].str.split('\s',n=1,expand= True)
        file['text'] = file['text'].str.lower()
        file = file.astype({'ID':int})
        file['text'] = file['text'].str.rstrip('. ! ?')
        file['story_num'] = None
        label += 1
        
        # 사전 구성
        word = pd.Series(file['text'].str.cat(sep = ' ').split())
        dictionary = dictionary.append(pd.Series(word.unique()))
        
        # 같은 스토리끼리 인덱스 부여(story_num)
        story_idx = file[file['ID'].isin(['1'])].index
        story_count = 1
        
        for i in story_idx:
            file.iloc[i,4] = story_count
            story_count += 1    
        file['story_num'].fillna(method='pad',inplace = True)
        
        # 질문과 답 찾기
        question = file[file['answer'].isna()==False]
        sup_facts_text = list()
        sup_facts_id = list()
        
        # supporting fact들 찾기(질문 별로) - 20개 한정(원래 논문에서도 최근 20개로 한정)
        for i, j in zip(question['ID'], question['story_num']):
            sup_fact = file[(file['ID'] < i) & (file['answer'].isna()) & (file['story_num']==j)]
            facts_text = sup_fact['text'].str.split().tolist()
            facts_text = facts_text[-20:]
            sup_facts_text.append(facts_text)
            facts_id = sup_fact['ID'].tolist()
            facts_id = facts_id[-20:]
            sup_facts_id.append(facts_id)
            
                        
        story = pd.DataFrame()
        story['question'] = question['text'].str.split()
		
		# task 별로 정답 개수가 다르기 때문에 label을 이용해서 처리 
		# task 8의 경우 최대 3개
        if label == 8:
            answer = file[file['answer'].isna()==False]['answer'].str.lower()\
                     .str.split(',',expand=True)    
            story['answer'] = answer.apply(lambda x: (x.dropna()).tolist(),axis = 1)
            dictionary = dictionary.append(pd.Series(answer[0].unique()))
            
        # task 19의 경우 2개 고정    
        elif label == 19:
            answer = file[file['answer'].isna()==False]['answer'].str.lower()\
                     .str.split(',',expand=True)    
            story['answer'] = answer.apply(lambda x: x.tolist(),axis = 1)
            dictionary = dictionary.append(pd.Series(answer[0].unique()))
            
        else:
            answer = file[file['answer'].isna()==False]['answer'].str.lower()
            story['answer'] = answer.apply(lambda x: [x])
            dictionary = dictionary.append(pd.Series(answer.unique()))
            
        story['sup_facts'] = sup_facts_text
        story['label'] = label
        story['sup_facts_id'] = sup_facts_id
        stories = stories.append(story)  
        
    stories.reset_index(inplace=True, drop=True)
    dictionary = dictionary.drop_duplicates().tolist()
    dictionary_size = len(dictionary)
        
    return stories, dictionary, stories['label']
# =============================================================================

# =============================================================================
# train, test 분리

def split_train_test(stories, label, perc):
    train, test = train_test_split(stories, test_size = perc, shuffle = True, stratify = label)
    
    train.reset_index(inplace = True, drop = True)
    test.reset_index(inplace = True, drop = True)
    
    return train, test
# =============================================================================

# =============================================================================
# dictionary 이용한 vector로 만든 후 tensor로 변환

def vectorize_stories(stories, dictionary, device):
    
    vectorize = []
    for i in stories.index:
        question = torch.tensor([dictionary.index(word) for word in stories.iloc[i,0]], device = device).long()
        answer = torch.tensor([dictionary.index(word) for word in stories.iloc[i,1]], device = device).long()
        facts = [torch.tensor([dictionary.index(word) for word in text], device = device).long() for text in stories.iloc[i, 2]]
        vectorize.append((question, answer, pad_sequence(facts, batch_first=True, padding_value=len(dictionary))))
        
    return vectorize
# =============================================================================

# =============================================================================
# DataLoader를 위한 Data Set 정의

class bAbidataset(Dataset):
    
    def __init__(self, stories):
        self.stories = stories
    
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, index):
        question, answer, facts = self.stories[index]
        
        return question, answer, facts
# =============================================================================

# =============================================================================
# collate 함수 정의
# DataLoader를 통해 batch를 만들 시에 task별로 텐서의 크기가 각각 다르기 때문에
# 하나로 통일시켜줘야 batch를 만들 수 있음
# question 차원: 배치 크기 * 질문 길이(단어 수)
# answer 차원: 배치 크기 * 답변 길이(단어 수)
# facts 차원: (배치 크기 * fact의 갯수) * 각 fact들의 단어 수: 2차원
# padding값은 사전 크기 + 1 -> embedding 고려

def batchify(data_batch):
    
    # padding (사전 크기) 사전 0 - 155
    # 나중에 보강
    pad = dictionary_size
    
    question_batch = []
    answer_batch = []
    facts_batch = []
    
    for batch in data_batch:
        question_batch.append(batch[0])
        answer_batch.append(batch[1])
        facts_batch.append(batch[2])
    
    question_batch = pad_sequence(question_batch, batch_first=True, padding_value=pad)
    answer_batch = pad_sequence(answer_batch, batch_first=True, padding_value=pad)
    
    # facts_batch 차원 설정
    row, column = max(t.size(0) for t in facts_batch), max(t.size(1) for t in facts_batch)
    ff = torch.ones(len(facts_batch), row, column, dtype=torch.long) * pad
    for i, t in enumerate(facts_batch):
        r, c = t.size(0), t.size(1)
        ff[i, :r, :c] = t
        
    return question_batch, answer_batch, ff.view(-1, ff.size(2))
# =============================================================================

# =============================================================================
# 정답 비교(정답률 계산)

def get_answer(predict, answer):
    
    with torch.no_grad():
        index = torch.argmax(predict, dim = 1)
        correct = (index == answer.squeeze()).sum().item()
        correct = (correct/float(predict.size(0)))
        
        return correct
# =============================================================================

