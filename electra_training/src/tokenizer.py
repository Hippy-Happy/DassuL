""" written by seoann 04/29/2022 """
import pandas as pd 
from transformers import ElectraTokenizerFast

class get_tokenizer():
    """ Get tokenizer, add tokens """

    def __init__(self):
        self.tokenizer = ElectraTokenizerFast.from_pretrained('kykim/electra-kor-base')
        token = pd.read_csv('./words/token.txt')
        self.tokenizer.add_tokens(str(token.columns[:]))