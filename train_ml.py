import pandas as pd
import numpy as np
from transformers import ElectraModel, ElectraTokenizer
from transformers import ElectraForSequenceClassification, AdamW
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
load = []


def Deserialization():
    models = ['sexual_minority', 
              'race', 
              'age_model',
              'local',
              'religion_model',
              'other',
              'badwords_koelectra_model',
              'clean_koelectra_model',
              'personal_koelectra_model',
              'gender_model'
              ]
    PATH = "C:/Users/ME/Desktop/venv/models/"
    for params in models:
        model = ElectraForSequenceClassification.from_pretrained('monologg/koelectra-small-v2-discriminator')
        model.classifier.out_proj =  nn.Sequential( nn.Linear(256, 1), nn.Sigmoid() )
        try:
            model.load_state_dict(torch.load(PATH + f'{params}.pth', map_location=device)['model_state_dict'])
        except:
            model.load_state_dict(torch.load(PATH + f'{params}.pth', map_location=device))
        load.append(model)
    print('model load clear')


def evaluate(data):
    

    md1 = load[0]
    md2 = load[1]
    md3 = load[2]
    md4 = load[3]
    md5 = load[4]
    md6 = load[5]
    md7 = load[6]
    md8 = load[7]
    md9 = load[8]
    md10 = load[9]

    predict_proba_df = pd.DataFrame()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(data):
        
        y_pred1 = md1(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred2 = md2(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred3 = md3(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred4 = md4(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred5 = md5(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred6 = md6(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred7 = md7(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred8 = md8(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred9 = md9(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
        y_pred10 = md10(input_ids_batch, attention_mask=attention_masks_batch)[0].tolist()
    
        tmp = pd.DataFrame([y_pred1,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6,y_pred7, y_pred8,y_pred9,y_pred10]).T
        predict_proba_df = pd.concat([predict_proba_df, tmp])
        

    predict_proba_df = predict_proba_df.reset_index(drop = True)
    predict_proba_df.columns = ['성소수자','인종국적','연령','지역','종교','기타혐오','악플욕설','clean','개인지칭','성별']
    print(predict_proba_df)      
    
    return predict_proba_df

