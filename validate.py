import torch
import torchvision.transforms
import numpy as np
from PIL import Image
from torch import nn
from load_LIDC_data import LIDC_IDRI

dataset = LIDC_IDRI(dataset_location = 'data/')
ind = 100
image = dataset[ind][1].numpy()
image = Image.fromarray(image)
image = image.convert('RGB')
print(image)


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128,128)),torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


class Model(nn.Module):
    def _init_(self):
        super(Model,self)._init_()
        self.model =  nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.model(x)
        return x


model = torch.load('model/net1.pt')
image = torch.reshape(image,(1,3,128,128))
model.eval()

with torch.no_grad():
    image = image.cuda()
    output = model(image)

print(output)
print(output.armax(1))