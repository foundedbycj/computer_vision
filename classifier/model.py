import torch.nn as nn
import torch

class NeuralNet(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.fc1 =nn.Sequential(
        nn.Conv2d(3,12,(2,2)),
        nn.ReLU(),
        nn.MaxPool2d((2,2),1),
        nn.Conv2d(12,24,(2,2)),
        nn.ReLU(),
        nn.MaxPool2d((2,2),1),
        nn.Conv2d(24,36,(3,3)),
        nn.ReLU(),
        nn.MaxPool2d((2,2),1),
        nn.Conv2d(36,64,(2,2)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(36864,2)

       
       
        
        )

    def forward(self,X):
      X = self.fc1(X)

      return X