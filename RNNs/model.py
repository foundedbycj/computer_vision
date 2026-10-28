import torch
import torch.nn as nn


class CustomRNN(nn.module):
    def __init__ (self):
        super().__init__()

        self.embedding = nn.Embedding(88587,64)


        self.rnn = nn.RNN(input_size=64,hidden_size=42,num_layers=1)

        self.fc = nn.Linear(32,1)

        self.sigmoid = nn.Sigmoid

    def forward(self,x):

        x =  self.embedding(x)
        output,h_f = self.rnn(x)
        x = h_f[-1]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x 
    



