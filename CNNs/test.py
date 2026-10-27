import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
from model import NeuralNet
from PIL import Image
import pandas as pd
from torch.nn import Softmax

class TestFolder (Dataset):
    def __init__ ( self,path,transform):
        self.path = path
        self.transform = transform
        self.filenames = [f for f in os.listdir(path) if f.lower().endswith('jpg')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path,self.filenames[idx])
        image = Image.open(img_path)
        image = self.transform(image)
                                
        return image

transform_ = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
])

test_folder = TestFolder(r'D:\computer_vision\DATASETS\test',transform=transform_)
test_loader = DataLoader(test_folder,batch_size=15,shuffle=False)


model = NeuralNet()
weights = torch.load(r'D:\computer_vision\CNNs\weights\saved_weights.pt' ,weights_only=True)
model.load_state_dict(weights)


# instantitate the softmax to bring the prediction --------

softmax  = Softmax(dim=1)

model.eval()
predictions = []
with torch.no_grad():
    for image in test_loader:
        result = model(image)
        prediction = softmax(result)
        predictions.extend(prediction.cpu().tolist())


data = pd.DataFrame(predictions)
print(data.round(4))
#------- Create the predicitions in a dataframe ----------
