""" written by seoann 04/29/2022 """

from transformers import ElectraTokenizerFast

class get_tokenizer():
    def __init__(self):
        self.tokenizer = ElectraTokenizerFast.from_pretrained('kykim/electra-kor-base')