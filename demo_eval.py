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
import sys

print(sys.argv[0])
print(sys.argv[1])
print(sys.argv[2])

data = pd.read_csv('/content/drive/MyDrive/aseml/train.csv')
data = data[["sourceLineTokens","targetLineTokens"]]
data=data.to_numpy()

listX = list(data[:,0])
listY = list(data[:,1])

line_len=70

for i in range(len(listX)):
    temp=["SOS"]
    listX[i]=ast.literal_eval(listX[i])
    temp.extend(listX[i])
    listX[i]=temp[0:line_len-1]
    listX[i].append("EOS")
    
for i in range(len(listY)):
    temp=["SOS"]
    listY[i]=ast.literal_eval(listY[i])
    temp.extend(listY[i])
    listY[i]=temp[0:line_len-1]
    listY[i].append("EOS")
    
    
vocabX={}
k=250
maxlenX=0;
for line in listX:
    
    
        
    for token in line:
        if(token not in vocabX.keys()):
            vocabX[token]=1
        else:
            vocabX[token]=vocabX[token]+1
    


sorted_vocabX = dict( sorted(vocabX.items(), key=operator.itemgetter(1),reverse=True))

top_namesX = dict(heapq.nlargest(k, vocabX.items(), key=itemgetter(1)))

index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3 : "OOV_Token"}
word2index  = {"PAD" : 0, "SOS" : 1,"EOS" : 2, "OOV_Token" : 3}

i = 4

for token in top_namesX.keys():
    if(token not in word2index) :
        index2word[i]=token
        word2index[token]=i
        i=i+1
    

valid_data = pd.read_csv("/content/drive/MyDrive/aseml/"+sys.argv[1])
data = valid_data[["sourceLineTokens","targetLineTokens", "targetLineText"]]
data=data.to_numpy()

latent_dim=256;
loaded_model = load_model('/content/drive/MyDrive/aseml/best.h5')

encoder_inputs = loaded_model.input[0]   
encoder_outputs, state_h_enc, state_c_enc = loaded_model.layers[2].output   
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = loaded_model.input[1]   
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = loaded_model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = loaded_model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

listX = list(data[:,0])
listY = list(data[:,1])

for i in range(len(listX)):
    temp=["SOS"]
    listX[i]=ast.literal_eval(listX[i])
    temp.extend(listX[i])
    listX[i]=temp[0:line_len-1]
    listX[i].append("EOS")
    
for i in range(len(listY)):
    temp=["SOS"]
    listY[i]=ast.literal_eval(listY[i])
    temp.extend(listY[i])
    listY[i]=temp[0:line_len-1]
    listY[i].append("EOS")


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

def decode_sequence(input_seq):
    
    states_value = encoder_model.predict(input_seq)

   
    target_seq = np.zeros((1, 1, k+2))
    
    target_seq[0, 0, word2index['SOS']] = 1.

    
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

      
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        
        sampled_char = index2word[sampled_token_index]
        
        decoded_sentence.append(sampled_char)

       
        if (sampled_char == 'EOS' or
           len(decoded_sentence) > line_len):
            stop_condition = True

        
        target_seq = np.zeros((1, 1, k+2))
        target_seq[0, 0, sampled_token_index] = 1.

       
        states_value = [h, c]
       

    return decoded_sentence

true=[]
pred=[]   

op=[]
for seq_index in range(len(listX)):
    input_seq = encoderX[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    op.append(decoded_sentence)
    true.append(data[seq_index,2])
    if("EOS" in decoded_sentence):
     decoded_sentence.remove("EOS")
    prediction=""
    for token in decoded_sentence:
        prediction=prediction+" "+token
    pred.append(prediction)
    print(seq_index)


valid_data['fixedTokens'] = op

valid_data.to_csv("/content/drive/MyDrive/aseml/"+sys.argv[2])
     
match=0;
for i in range(len(listX)):
  if((true[i].strip()==pred[i].strip())):
    match=match+1
match


