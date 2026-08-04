import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms


transform_train = transforms.Compose(
    [transforms.RandomResizedCrop((256,256)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[])]
)