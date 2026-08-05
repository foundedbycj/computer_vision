import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder


transform_train = transforms.Compose(
    [transforms.RandomResizedCrop((256,256)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[485,456,402],std=[229,256,210])])

transform_test = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[485,456,402],std=[229,256,210])]
)


train_data = ImageFolder(r'D:\computer_vision\DATASETS',transform_train)
test_data = ImageFolder(r"D:\computer_vision\DATASETS",transform_test)


print(cat)