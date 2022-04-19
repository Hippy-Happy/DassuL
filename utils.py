# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:26:33 2022

@author: ME
"""

import unicodedata
from transformers import ElectraModel, ElectraTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LoadDataset(Dataset):
    def __init__(self, df, tk):
        self.df = df
        self.tokenizer = tk
        
    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        row = self.df.iloc[idx, :].values
        # target이 없는경우 (즉, 문장만 입력된 경우)
        if len(row) <= 1:
            text = row[0]

            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=50,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            return input_ids, attention_mask     
            
        # target이 있는 경우 (원래 코드)
        else:
            text = row[0]
            y = row[1]

            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=50,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            return input_ids, attention_mask, y
    
def clean_text(text, tokenizer):
    sentence_df = pd.DataFrame([text], columns=['문장'])
    sentence_df[['target', 'target2', 'target3', 'target4', 'target5']] = 1, 1, 1, 1, 1
    sentence_df[['target6', 'target7', 'target8', 'target9', 'target10']] = 1, 1, 1, 1, 1
    inputs = LoadDataset(sentence_df, tokenizer)
    inputs_loader = DataLoader(inputs, batch_size=1)
    return inputs_loader

def find_label(data):
    labels = ''
    for col in range(0, 10):
        if data.iloc[0,col] >= [0.8]:
            labels = labels + str(col)
            if col == 7:
                labels = ''
                break
    labels = labels.replace("0", "성소수자 ")
    labels = labels.replace("1", "인종/국적 ")
    labels = labels.replace("2", "연령 ")
    labels = labels.replace("3", "지역 ")
    labels = labels.replace("4", "종교 ")
    labels = labels.replace("5", "기타 ")
    labels = labels.replace("6", "악플/욕설 ")
    labels = labels.replace("8", "개인지칭 ")
    labels = labels.replace("9", "성별 ")
    labels = labels + "비하 표현이 있습니다."
    
    if labels == "비하 표현이 있습니다.":
      labels = ' 이 문장은 깨끗합니다! '
    print('*********************************************************************************************')
    return labels