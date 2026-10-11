import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 3 × H × W
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 16 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 32 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 64 channel 
            nn.Conv2d(64,128,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            nn.Flatten())

        self.classifier = nn.Sequential(
            nn.Linear(128,60),
            nn.Dropout(),
            nn.Linear(60,2)
        )

     

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x