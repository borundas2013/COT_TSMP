# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:04:53 2021

@author: Cheng
"""
from itertools import combinations
import numpy as np
import itertools
import string
import csv
from tensorflow.keras.models import Sequential,load_model
import os
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D,GlobalMaxPool1D,Dropout,BatchNormalization,Dense
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import pandas as pd
import pandas as pd

df=pd.read_excel('smiles.xlsx')
smiles_total=df['SMILES'][0:2]

#delete white space in the smiles
i=0
for i in range(0,int(len(smiles_total))):
          smiles_total[i]=smiles_total[i].translate({ord(c): None for c in string.whitespace})
          smiles_total[i] = smiles_total[i].split(',')
          smiles_total[i]=[item.replace("{", "") for item in smiles_total[i]]
          smiles_total[i]=[item.replace("}", "") for item in smiles_total[i]]

# smiles=pd.Series(smiles)          
smiles_total1=smiles_total 
need_no=1
combine_total=[]        

              
smiles_1=[[] for i in range(int(len(smiles_total)))]
smiles_2=[[] for i in range(int(len(smiles_total)))]
smiles_3=[[] for i in range(int(len(smiles_total)))]
smiles_4=[[] for i in range(int(len(smiles_total)))]
i=0
for i in range(0,int(len(smiles_total))):
    smiles_1[i]=smiles_total[i][0]
    if len(smiles_total[i])>1:
        smiles_2[i]=smiles_total[i][1]


smiles_to_latent_model=load_model("Blog_simple_smi2lat7_150")
latent_to_states_model=load_model("Blog_simple_lat7state7_150")
sample_model=load_model("Blog_simple_samplemodel7_150")

#predict the new materials
def vector_to_smiles(X):#(latent):
    X=X.reshape(1,X.shape[0],X.shape[1],1)
    x_latent = smiles_to_latent_model.predict(X)
    #decode states and set Reset the LSTM cells with them
    states = latent_to_states_model.predict(x_latent)
    sample_model.layers[1].reset_states(states=[states[0],states[1]])
    #Prepare the input char
    startidx = char_to_int["!"]
    samplevec = np.zeros((1,1,len(charset)))
    samplevec[0,0,startidx] = 1
    smiles = ""
    #Loop and predict next char
    for i in range(205):
        o = sample_model.predict(samplevec)
        sampleidx = np.argmax(o)
        samplechar = int_to_char[sampleidx]
        if samplechar != "E":
            smiles = smiles + int_to_char[sampleidx]
            samplevec = np.zeros((1,1,len(charset)))
            samplevec[0,0,sampleidx] = 1
        else:
            break
    return x_latent
def split1(word):
    return [char for char in word]
def vectorize1(smiles):
        smiles=split1(smiles)
        one_hot =  np.zeros(( embed-1 , len(charset)),dtype=np.int8)
        # for i,smile in enumerate(smiles):
            #encode the startchar
        one_hot[0,char_to_int["!"]] = 1
            #encode the rest of the chars
        for j,c in enumerate(smiles):
                one_hot[j+1,char_to_int[c]] = 1
            #Encode endchar
        one_hot[len(smiles)+1:,char_to_int["E"]] = 1
        #Return two, one for input and the other for output
        return one_hot#[:,0:-1,:]#, one_hot[:,1:,:]
charset=['-', 'F', 'S', '9', 'N', '(', 'l', 'P', 'L', 'T', 'p', 'r', 'A', 'K', 't', ']', '1', 'X', 'R', 'o', '!', 'c', '#', 'C', '+', 'B', 's', 'a', 'H', '8', 'n', '6', '4', '[', '3', ')', '0', '%', 'i', '.', '=', 'g', 'O', 'Z', 'E', '/', '@', 'e', '\\', 'I', 'b', '7', '2', 'M', '5']
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
latent_dim=256;embed=205

combined_vetor_all=np.zeros((int(len(smiles_total)),1,latent_dim))



Neutral_model = Sequential()

Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(256,1)))
Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

# Neutral_model.add(MaxPooling1D(pool_size=2))
Neutral_model.add(GlobalMaxPool1D())

# Neutral_model.add(Flatten())
Neutral_model.add(BatchNormalization())
Dropout(0.4)
Neutral_model.add(Dense(256, activation='relu'))

Neutral_model.add(Dense(64, activation="relu"))
# Dropout(0.4)
Neutral_model.add(Dense(64, activation="relu"))
Neutral_model.add(Dense(64, activation="relu"))
Neutral_model.add(Dense(32, activation="relu"))
Neutral_model.add(Dense(32, activation="relu"))
# check to see if the regression node should be added
Neutral_model.add(Dense(1, activation="linear"))

def root_mean_squared_error(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true)/K.abs(y_true))
Neutral_model.compile(loss="mae", optimizer='adam',metrics=[root_mean_squared_error])#[tf.keras.metrics.MeanSquaredError()])

molar_ratio=0.0

for n in range(2):
    molar_ratio=molar_ratio+0.1
    print(len(smiles_total))
    # pred_Tg = np.ones(len(smiles_total), dtype=np.float32)
    # pred_Er = np.ones(len(smiles_total), dtype=np.float32)
    for i in range(0,int(len(smiles_total))):
        vec1=vectorize1(smiles_1[i])
        latent_v1=vector_to_smiles(vec1)
        combined_vetor=latent_v1*molar_ratio
        if len(smiles_total[i])>1:
            vec2=vectorize1(smiles_2[i])
            latent_v2=vector_to_smiles(vec2)
            combined_vetor=latent_v1*molar_ratio+latent_v2*(1-molar_ratio)
        combined_vetor_all[i]=combined_vetor


    Neutral_model.load_weights('conv1d_model1_Tg245_3.h5')


    j=len(smiles_total)
    pred_Tg=np.ones(j, dtype=np.float32)
    for k in range(0,j):
        to_predict=combined_vetor_all[k].reshape(1,256,1)
        pred_Tg[k]=Neutral_model.predict(to_predict)



    Neutral_model.load_weights('conv1d_model1_Er245_2.h5')
    pred_Er=np.ones(j, dtype=np.float32)
    for k in range(0,j):
        to_predict=combined_vetor_all[k].reshape(1,256,1)
        pred_Er[k]=Neutral_model.predict(to_predict)

                # looking for the suitable SMP
    good_sample=[]
    for i in range(0,len(smiles_total)):
        good_sample.append(i)
                      #if abs(pred_Er[i]-350)/350<0.10 and pred_Tg[i]<280*0.9:
                       #   good_sample.append(i)
                #print(good_sample)

    selected_smiles = [smiles_total[index] for index in good_sample]
    selected_Er = [pred_Er[index] for index in good_sample]
    selected_Tg = [pred_Tg[index] for index in good_sample]



    molar_ratio1=np.ones(len(selected_smiles))*molar_ratio
    molar_ratio2=np.ones(len(selected_smiles))*(1-molar_ratio)
    rows = zip(selected_smiles,molar_ratio1,molar_ratio2,selected_Er,selected_Tg)



    with open('good sample1 ALL.csv', 'a+',newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
