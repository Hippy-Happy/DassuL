""" written by seoann 04/29/2022 """

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

class evaluate():
    """ evaluating test data """

    def __init__(self, model, batch_size, test_set, category, device):
        
        self.model = model
        self.test_set = test_set
        self.category = category
        self.device = device
        self.batch_size = batch_size

        self.load_dataset()
        self.evaluate_test_data()

    def load_dataset(self):
        """ Data Load """
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)
    
    def evaluate_test_data(self):
        
        self.model.eval()
        score_list = []
        
        for input_ids_batch, attention_masks_batch, y_batch in tqdm(self.test_loader):
            
            input_ids_batch = input_ids_batch.to(self.device)
            attention_masks_batch = attention_masks_batch.to(self.device)
            
            y_batch = y_batch.to(self.device)
            y_pred = self.model(input_ids_batch, attention_mask=attention_masks_batch).logits.reshape(-1)
       
            try:
                score = roc_auc_score(y_batch.tolist(), y_pred.tolist())
                score_list.append(score)
            except: 
                pass
        
        # print score
        print("**************************")
        print(self.category)
        print("epoch roc_auc:", np.mean(score_list))