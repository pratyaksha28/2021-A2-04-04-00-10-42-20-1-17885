from keras.models import load_model

from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import *
from keras.initializers import *
import tensorflow as tf
import time, random
import pandas as pd
import ast
import heapq
import operator
from operator import itemgetter
import numpy as np
from keras.optimizers import Adam
from keras.layers import Embedding
from tensorflow import keras

data = pd.read_csv('train.csv')
data = data[["sourceLineTokens","targetLineTokens"]]
data=data.to_numpy()

listX = list(data[:,0])
listY = list(data[:,1])

line_len=25

for i in range(len(listX)):
    start=["SOS"]
    listX[i]=ast.literal_eval(listX[i])
    start.extend(listX[i])
    listX[i]=start[0:line_len-1]
    listX[i].append("EOS")
    
for i in range(len(listY)):
    start=["SOS"]
    listY[i]=ast.literal_eval(listY[i])
    start.extend(listY[i])
    listY[i]=start[0:line_len-1]
    listY[i].append("EOS")
    
    
vocab={}
k=250
maxlen=0;
for line in listX:               
    for token in line:
        if(token not in vocab.keys()):
            vocab[token]=1
        else:
            vocab[token]=vocab[token]+1
    


sorted_vocab = dict( sorted(vocab.items(), key=operator.itemgetter(1),reverse=True))
top_names = dict(heapq.nlargest(k, vocab.items(), key=itemgetter(1)))

index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3 : "OOV_Token"}
word2index  = {"PAD" : 0, "SOS" : 1,"EOS" : 2, "OOV_Token" : 3}

i = 4

for token in top_names.keys():
    if(token not in word2index) :
        index2word[i]=token
        word2index[token]=i
        i=i+1
       
input_seq  = np.zeros((len(listX) ,  line_len),dtype='int64')
for i in range(len(listX)):
    for j in range(len(listX[i])):
      if listX[i][j] in word2index.keys():
        input_seq[i , j] =word2index[listX[i][j]]
      else:
        input_seq[i , j] = 3

encoderX = np.zeros((len(listX) , line_len , k+2 ))
for i in range(input_seq.shape[0]):
  for j in range(line_len):
    encoderX[i , j , input_seq[i , j]] = 1
  
output_seq = np.zeros((len(listY) , line_len),dtype='int64')
for i in range(len(listY)):
    for j in range(len(listY[i])):
        if listY[i][j] in word2index.keys():
            output_seq[i , j] =word2index[listY[i][j]]
        else:
            output_seq[i , j] = 3
            
decoderX = np.zeros((len(listY) , line_len , k+2 ))
decoderY = np.zeros((len(listY) , line_len , k+2 ))
for i in range(output_seq.shape[0]):
  for j in range(line_len):
    decoderX[i , j , output_seq[i , j]] = 1
    if j > 0:
      decoderY[i , j-1 , output_seq[i , j]] = 1


batch_size = 128
latent_dim = 50
 

encoder_inputs = Input(shape=(None, k+2))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, k+2))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(k+2, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)   

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['acc'])

model.fit([encoderX, decoderX], decoderY, batch_size = batch_size, epochs=50, validation_split=0.2)
      
model.save('best.h5')