import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
from model import NeuralNet 
import matplotlib.pyplot as plt

# make up a transform to alter the image for better processing
transform_ = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
])

train_data = ImageFolder(r'D:\computer_vision\DATASETS\train',transform=transform_) 

train_loader=DataLoader(dataset=train_data,batch_size=32,shuffle=True,num_workers=0) # make a dataloader to pack up the data folder and separate them into images and labels
 
model = NeuralNet() #instantiate the model from 'model.py' 

criterion = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(),lr = 0.0005)# set up the optimiser


epoch = 40
losses  = []
for epochs in range (epoch):
    running_loss = 0.0

    for images, label in train_loader:

        optimiser.zero_grad()
        output = model(images)


        loss = criterion(output,label) # compare the outputs (predicted vs actual)
        loss.backward() #backpropagation
        optimiser.step()

        running_loss += loss.item()

    epoch_loss = running_loss/len(train_loader)
    losses.append(epoch_loss) # append the loss at each epoch so as to plot them into the graph
    print(f'loss at epoch {epochs+1} is {epoch_loss} ') # check the loss at each instance of epoch



   

n_epoch = list(range(1,epoch+1)) # print the 



# visualise the loss curve
plt.plot(n_epoch,losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


torch.save(model.state_dict(),'/classifier/saved_weights.pt')