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

#split df_train and y_train into 5
df1, df2, df3, df4, df5 = np.array_split(df_train, 5)
y1, y2, y3, y4, y5 = np.array_split(y_train, 5)

learning_rate = 0.01
epochs = 3000
for neurons in [10]:
    SEED=42
    torch.manual_seed(SEED) #seed for random initialization of weights

    #DEFINE NEURAL NETWORK
    class NN_HOMO(nn.Module):
    
        def __init__(self, input_size):
            super(NN_HOMO, self).__init__()
            print(neurons)
            self.input_layer = nn.Linear(input_size,neurons)
            self.hidden_layer_01 = nn.Linear(neurons,neurons)
            self.hidden_layer_02 = nn.Linear(neurons,neurons)
            self.out_layer = nn.Linear(neurons,1)
    
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) #parameter optimizer model, Stochastic Gradient Descent  
        for epoch in range(max_epoch):  #loop over the training dataset multiple times
        
            yt_pred = model(Xt_tensor)
        
            loss = criterion(yt_pred, yt_tensor) #compute the loss
        
            optimizer.zero_grad() #zero the gradients
        
            loss.backward() #perform a backward pass (backpropagation)
            optimizer.step() #update the parameters


    #TRAIN AND TEST NEURAL NETWORK
    input_size = len(df_train.columns) #length of feature vector = number of input nodes in the neural network

    model = NN_HOMO(input_size)
    pytorch_total_params = sum(p.numel() for p in model.parameters()) #number of parameters in model

    #TRAIN and VAL step
    lr=learning_rate
    max_epoch=epochs
    skip = 0
    RMSEs = []

    #5-fold cross-validation
    for i, j in zip([df1, df2, df3, df4, df5],[y1, y2, y3, y4, y5]): #loop parallel
        frames = [df1, df2, df3, df4, df5]
        del frames[skip]
        df_train = pd.concat(frames)
    
        ys = [y1, y2, y3, y4, y5] 
        del ys[skip]
        y_train = pd.concat(ys)

        skip = skip + 1
    
        model.train() #model in train mode
        train_nn(model,df_train,y_train, lr, max_epoch)

        #EVAL step
        model.eval() #model in eval mode

        df_val_tensor = torch.tensor(i.values) #change dataframe to tensor
        df_val_tensor = df_val_tensor.type(torch.float32) #change type of tensor to float

        with torch.no_grad():
            y_predicted_val = model(df_val_tensor) #predicted HOMO energies
    
        y_predicted_val = y_predicted_val.numpy()
        y_predicted_val = np.ndarray.reshape(y_predicted_val,(1,j.shape[0]))
        y_predicted_val = np.ndarray.reshape(y_predicted_val,(j.shape[0],))


        RMSE = np.sqrt(mean_squared_error(j, y_predicted_val))
        print("RMSE is "+str(RMSE)+" for part "+str(skip)+" with 3 hidden layers and "+str(neurons)+" neurons.")
        RMSEs.append(RMSE)

    from statistics import mean
    print("Mean val RMSE is "+str(mean(RMSEs)))


