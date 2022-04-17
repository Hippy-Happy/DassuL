"""
Created on Fri Apr 15 17:20:16 2022

@author: seoann
"""
from flask import Flask, request, jsonify
from model import ModelHandler
from transformers import ElectraModel, ElectraTokenizer
from transformers import ElectraForSequenceClassification, AdamW
import warnings


app = Flask (__name__)
handler = ModelHandler()
warnings.filterwarnings('ignore')


@app.route('/')
def chatbot():
    return 'Chat bot Server'


@app.before_first_request
def before_first_request():
    handler.initialize()
    
    

@app.route('/pred', methods = ['POST'])
def pred():
    body = request.get_json()
    text = body['text']
    print(text)
    output = handler.handle(text)
    
    #DB 저장
    """
    text와 user info를 전송
    """
    return jsonify(output)
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = '5000', debug = True)
