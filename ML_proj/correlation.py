# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:56:00 2019

@author: jo345
"""

import pandas as pd
import numpy as np
import torch

data_x = torch.load('tensorx.pt')
data_y = torch.load('tensory.pt')

data_x = data_x.numpy()
data_y = data_y.numpy()

df_x = pd.DataFrame(data_x)
corr = df_x.corr(method = 'pearson')
print(corr)

np.savetxt('correlation.txt', corr)