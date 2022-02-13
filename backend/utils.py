# from flask import Flask, render_template, request, render_template_string

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model

# import transformers
# from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
# from transformers import DistilBertModel, DistilBertTokenizer
# import torch
# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from textwrap import wrap
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import roc_auc_score

import tensorflow_hub as hub

np.random.seed(0)

# if(0):
train = pd.read_csv('train.csv')
cv = pd.read_csv('cv.csv')
train = pd.concat([train, cv], ignore_index=True)

train_text = (train['book_title'].map(str) + ' ~~~ ' + train['sentence'].map(str)).to_numpy()
train_labels = train['sent_spoil'].to_numpy().astype(np.int32)


reviewMaxLen = 600

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(train_text)

model = load_model('model')

def spoiler_value(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=reviewMaxLen)
    prediction = model.predict(pad).item()
    return(prediction) 
# else:
#     PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
#     class SpoilerClassifier(nn.Module):
#         def __init__(self, n_classes):
#             super(SpoilerClassifier, self).__init__()
#             self.bert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
#             # self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
#             self.drop = nn.Dropout(p=0.3)
#             self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
#         def forward(self, input_ids, attention_mask):
#             model_output = self.bert(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask
#             )
#             last_hidden_state = model_output[0]
#             pooled_output = last_hidden_state[:, 0, :]
#             output = self.drop(pooled_output)
#             return self.out(output)

#     model = torch.load('best_model_state.bin') 

