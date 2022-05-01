""" written by seoann 04/29/2022 """

import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch import torch
from sklearn.model_selection import train_test_split




class LoadDataset(Dataset):
    """ load dataset """
    def __init__(self, df, tk, device):
        self.df = df
        self.tokenizer = tk
        self.device = device

    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, idx):
        row = self.df.iloc[idx, :].values
        if len(row) <= 1:
            text = row[0]

            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=128,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            
            input_ids = inputs['input_ids'][0].to(self.device)
            attention_mask = inputs['attention_mask'][0].to(self.device)

            return input_ids, attention_mask     
            
        else:
            text = row[0]
            y = row[1]

            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=128,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            
            input_ids = inputs['input_ids'][0].to(self.device)
            attention_mask = inputs['attention_mask'][0].to(self.device)

            return input_ids, attention_mask, y

class split_data():
    """ data split for training """

    def dataSplit(dataset, y_label):
        X_train, X_val = train_test_split(dataset, test_size = 0.2, stratify = dataset[y_label], random_state = 427)
        return X_train, X_val

