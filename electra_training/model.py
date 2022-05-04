""" written by seoann 04/29/2022 """

from transformers import ElectraForSequenceClassification
import torch
from torch import nn


class get_pretrained_model():
    """ get pretrained model, set out_proj """

    def __init__(self):
        self.model = ElectraForSequenceClassification.from_pretrained('kykim/electra-kor-base')
        self.model.classifier.out_proj =  nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())
        print(self.model.parameters())
        

    
    def set_device(self):   
        """ set model device as cuda """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
