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

test_data = ImageFolder(r'D:\computer_vision\DATASETS\test',transform=transform_)
test_loader=DataLoader(dataset=test_data,batch_size=8,shuffle=False,num_workers=0)



model = NeuralNet()

for image,label in test_loader:
    release = model(image)

print(release.shape)