# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 01:56:30 2019

@author: jo345
"""

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

    
data_x = torch.load('tensorx.pt')
data_y = torch.load('tensory.pt')

data_x = data_x.float()
data_y = data_y.float()

data_y_mean = torch.mean(data_y, dim=1)
#print(data_y)
    
#print(len(x_pca))
#673

#training_set: 623, testing_set: 50

train_data_x = data_x[:623]
#train_data_y = data_y_mean[:623]
train_data_y = data_y[:623]
#print(len(train_data_x))
test_data_x = data_x[623:]
#test_data_y = data_y_mean[623:]
test_data_y = data_y[623:]
#print(len(test_data_x))

X = scale(train_data_x.numpy())

pca = PCA(n_components = 26)

pca.fit(X)

var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

print(var1)
plt.plot(var1)
plt.show()

#lookig at plot, we have to use all of 26 features.
#we should not use pca

inputs = train_data_x
targets = train_data_y
#define model
model = nn.Linear(26,2)
opt = torch.optim.SGD(model.parameters(), lr = 1e-5)
loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)

train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(train_ds, shuffle=True)

def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))

#train for 100 epoch    
fit(100, model, loss_fn, opt)

#test
test_loss = loss_fn(model(test_data_x), test_data_y)
print('Testing loss: ', test_loss)
