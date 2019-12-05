# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:45:42 2019

@author: jo345
"""

import torch
import matplotlib.pyplot as plt

data_x = torch.load('tensorx.pt')
data_y = torch.load('tensory.pt')

data_x = data_x.float()
data_y = data_y.float()

def PCA(data_x, k=2):
    
    x_mean = torch.mean(data_x, 0)
    data_x = data_x - x_mean.expand_as(data_x)

    U, S, V = torch.svd(torch.t(data_x))
    C = torch.mm(data_x,U[:,:k])
    return C

y_sum = torch.mean(torch.t(data_y), 0)
t_alc = torch.t(data_y)
d_alc = t_alc[0]
w_alc = t_alc[1]
#print(y_sum)
#print(len(y_sum))
x_pca = PCA(data_x)

#print(x_pca)
#print(len(x_pca))

#print(torch.t(x_pca))
scatter_x = torch.t(x_pca)[0]
#print(max(scatter_x))
#print(min(scatter_x))
scatter_y = torch.t(x_pca)[1]
#print(max(scatter_y))
#print(min(scatter_y))
#print(scatter_x)
#1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5 =9종류

#plt.scatter(scatter_x, scatter_y)

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#000000']

#Dalc + Walc/2
for i in range(11):
    plt.scatter(scatter_x[y_sum == i/2], scatter_y[y_sum == i/2], c=colors[i], s=5, label=i)
    plt.show()
"""
for i in range(5):
    plt.scatter(scatter_x[d_alc == i], scatter_y[d_alc == i], c = colors[i], s=5, label=i)
plt.show()

for i in range(5):
    plt.scatter(scatter_x[w_alc == i], scatter_y[w_alc == i], c = colors[i], s=5, label=i)
plt.show()
"""

print('no meaningful cluster spotted when spreaded in 2d')
print('dimensionality reduction method: PCA')

