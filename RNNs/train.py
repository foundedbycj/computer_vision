# ---------------------------IMPORTS ----------------------
import numpy as np
import torch
import torch.nn
import pandas as pd

from keras.datasets import imdb
from torch.nn.utils.rnn import pad_sequence
from keras.utils import pad_sequences
from torch.nn import Sequential
import torch.nn as nn

#-----------------------Load & Preprocessing -------------

(X_train,y_train) , (X_test,y_test) = imdb.load_data()

X_train = pad_sequences(X_train,padding='post')
X_test = pad_sequences(X_test,padding='post')

X_train =  torch.tensor(X_train)
X_test  =  torch.tensor(X_test)
y_train =  torch.tensor(y_train)
X_test =  torch.tensor(y_test)


print(X_train.max())
