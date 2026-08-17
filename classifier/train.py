import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
from model import NeuralNet 


transform_ = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
])

train_data = ImageFolder(r'D:\computer_vision\DATASETS\train',transform=transform_)

train_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=False,num_workers=0)

model = NeuralNet()

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),lr = 0.00002)


epoch = 20

for epochs in range (epoch):
   

   for images,labels in train_loader:
       optimiser.zero_grad()

       output = model(images)    
    
       error = criterion(output,labels)

       

       error.backward()

       optimiser.step()
   print(f"loss of model at epoch {epochs+1}/{epoch} is {error}.4f")
 
torch.save(model.state_dict(), "cat_dog_model.pth")
