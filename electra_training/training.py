""" written by seoann 04/29/2022 """

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AdamW 
from transformers.optimization import get_cosine_schedule_with_warmup
from torch import nn
import torch
from tqdm.notebook import tqdm
import numpy as np
from torch.nn import functional as F
from early_stopping import EarlyStopping


class train_model():
    """ Train and evaluate model """

    def __init__(self, model, epochs, batch_size, train_set, validation_set, category, device):
        """ Initialize, set parameters and optimizer """
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.category = category
        self.device = device

        self.train_set = train_set
        self.validation_set = validation_set
        self.load_dataset()
        self.train(model)


    def load_dataset(self):
        """ Data Load """

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size)
        self.validation_loader = DataLoader(self.validation_set, batch_size=self.batch_size)



    def train(self, model):
        """ train and evaluate """
        
        warmup_ratio = 0.1
        t_total = len(self.train_set) * (self.epochs)
        optimizer = AdamW(self.model.parameters(), lr=1e-5, eps = 1e-8)
        loss_f = nn.BCEWithLogitsLoss()
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1 , num_training_steps=t_total)

        early_stopping = EarlyStopping(7, True, 0, self.category)

        for i in tqdm(range(self.epochs)):
            
            train_loss_list = [] 
            val_loss_list = []
            val_score_list = []

            epoch_train_loss = []
            epoch_val_loss = []
            epoch_val_score = []
            
            # train
            model.train()
                        
            
            for input_ids_batch, attention_masks_batch, y_batch in self.train_loader:


                input_ids_batch = input_ids_batch.to(self.device)
                attention_masks_batch = attention_masks_batch.to(self.device)
                y_batch = y_batch.to(self.device)           

                optimizer.zero_grad()
                y_pred = model(input_ids_batch, attention_mask=attention_masks_batch).logits.reshape(-1)
                loss = loss_f(y_pred.type(torch.FloatTensor), y_batch.type(torch.FloatTensor))

                loss.backward()               
                optimizer.step()
                scheduler.step()

                train_loss_list.append(loss.item())



            # validation loss
            model.eval()
            
            for input_ids_batch_val, attention_masks_batch_val, y_batch_val in self.validation_loader:
                
                input_ids_batch_val = input_ids_batch_val.to(self.device)
                attention_masks_batch_val = attention_masks_batch_val.to(self.device)
                y_batch_val = y_batch_val.to(self.device)
                
                y_pred_val = model(input_ids_batch_val, attention_mask = attention_masks_batch_val).logits.reshape(-1)
                loss = loss_f(y_pred_val.type(torch.FloatTensor), y_batch_val.type(torch.FloatTensor))
                
                val_score = roc_auc_score(y_batch_val.tolist(), y_pred_val.tolist())
                val_loss_list.append(loss.item())
                val_score_list.append(val_score)



            # calculate loss per epoch for early stopping
            train_loss = np.average(train_loss_list)
            val_loss = np.average(val_loss_list)
            val_score = np.average(val_score_list)

            epoch_train_loss.append(train_loss)
            epoch_val_loss.append(val_loss)
            epoch_val_score.append(val_score)
            epoch_len = len(str(self.epochs))

            print_msg = (f'[{i:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {val_loss:.5f} ' +
                        f'valid_score: {val_score:.5f}')

            print(print_msg)
    


            # clear lists to track next epoch
            train_loss_list = []
            val_loss_list = []
            val_score_list = []
            
            early_stopping(val_loss, self.model)
            
            if early_stopping.early_stop:
                print('early stopping')
                break
        # save checkpoint file
        model.load_state_dict(torch.load(f'./src/checkpoint/checkpoint_{self.category}.pt', map_location=self.device))