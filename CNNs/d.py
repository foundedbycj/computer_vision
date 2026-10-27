from PIL import Image
from torchvision import transforms
from CNNs.classifier.model import NeuralNet
image = Image.open('DATASETS/test/0001_dog_dog_19.jpg')
transform_ = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
])


image = transform_(image)



model = NeuralNet()
print(model(image).shape)