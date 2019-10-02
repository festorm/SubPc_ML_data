#!/home/koerstz/anaconda3/envs/quantum_ml/bin/python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy
sns.set()

from collections import namedtuple
import copy

import pickle

import math
from numbers import Number

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

torch.set_num_threads(1)

#LOAD DATABASES 
one_hot = pd.read_pickle("one_hot_no_ttf.pkl")
one_hot.index = range(0, len(one_hot))

df_train = pd.read_pickle("df_train_seed42.pkl")
y_train = pd.read_pickle("y_train_seed42.pkl")

df_test = pd.read_pickle("df_test_seed42.pkl")
y_test = pd.read_pickle("y_test_seed42.pkl")

for wd in [0.001]:
    SEED=42
    torch.manual_seed(SEED)    

    #DEFINE NEURAL NETWORK
    class NN_HOMO(nn.Module):
    
        def __init__(self, input_size):
            super(NN_HOMO, self).__init__()
            self.input_layer = nn.Linear(input_size,10)
            self.hidden_layer_01 = nn.Linear(10,10)
            self.hidden_layer_02 = nn.Linear(10,10)
            self.out_layer = nn.Linear(10,1)
    
        def forward(self, x):
            x = F.relu(self.input_layer(x))
            x = F.relu(self.hidden_layer_01(x))
            x = F.relu(self.hidden_layer_02(x))
            x = self.out_layer(x)       
        
            return x


    #DEFINE TRAINING LOOP FOR NEURAL NETWORK
    def train_nn(NN, Xtrain, ytrain, learning_rate, max_epoch):
    
        # Create PyTorch tensors
        Xt_tensor = torch.tensor(Xtrain.values) #change dataframe to tensor
        Xt_tensor = Xt_tensor.type(torch.float32) #change type of tensor to float

        yt_tensor = torch.tensor(ytrain.values)
        yt_tensor = yt_tensor.type(torch.float32)
        yt_tensor = torch.transpose(yt_tensor.unsqueeze(0),0,1)

        # train model
        model = NN
    
        criterion = nn.MSELoss() # Mean Squared Loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd) #parameter optimizer model, Stochastic Gradient Descent  
        for epoch in range(max_epoch):  #loop over the training dataset multiple times
        
            yt_pred = model(Xt_tensor)
        
            loss = criterion(yt_pred, yt_tensor) #compute the loss
            #print('epoch: ', epoch,'loss: ', loss.item()) #print the loss
        
            optimizer.zero_grad() #zero the gradients
        
            loss.backward() #perform a backward pass (backpropagation)
            optimizer.step() #update the parameters


    #TRAIN AND TEST NEURAL NETWORK
    input_size = len(df_train.columns) #length of feature vector = number of input nodes in the neural network

    model = NN_HOMO(input_size)
    pytorch_total_params = sum(p.numel() for p in model.parameters()) #number of parameters in model

    #TRAIN and VAL step
    lr=0.01
    max_epoch=3000
    
    model.train() #model in train mode
    train_nn(model,df_train,y_train, lr, max_epoch)

    #EVAL step
    model.eval() #model in eval mode

    df_test_tensor = torch.tensor(df_test.values) #change dataframe to tensor
    df_test_tensor = df_test_tensor.type(torch.float32) #change type of tensor to float

    with torch.no_grad():
        y_predicted = model(df_test_tensor) #predicted HOMO energies
    
    y_predicted = y_predicted.numpy()
    y_predicted = np.ndarray.reshape(y_predicted,(1,2268))
    y_predicted = np.ndarray.reshape(y_predicted,(2268,))


    RMSE = np.sqrt(mean_squared_error(y_test, y_predicted))
    print("RMSE is "+str(RMSE)+"for weight decay "+str(wd)+" with 3 hlayers, 10 hnodes, lr 0.01, and 3000 epochs")

#SAVE DATA
#save predicted HOMO energies for the test data
np.savetxt("estate_HOMO_predicted_wd0001_hl3_hn10_lr001_3000e.csv",y_predicted, delimiter=",")

#save calculated HOMO energies for the test data
np.savetxt("estate_HOMO_calculated_wd0001_hl3_hn10_lr001_3000e.csv",y_test.values,delimiter=",")

#save trained model as a pickle file
torch.save(model, "NN_estate_HOMO_wd0001_hl3_hn10_lr001_3000e.pkl", pickle_module=pickle)

