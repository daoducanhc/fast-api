from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

input_size = (100,100)

tol_mean = [0.5686043, 0.47916356, 0.44574347]
tol_std = [0.28832513, 0.26202822, 0.2570394 ]

transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor(),
                                transforms.Normalize(tol_mean, tol_std)
                                ])

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 10 * 10, 250)
        self.fc2 = nn.Linear(250, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = x.view(-1, 64 * 10 * 10)
        x = F.dropout(F.relu(self.fc1(x)), p=0.4)
        x = self.fc2(x)

        return F.softmax(x, dim=1)

def load_model():
    model = Net(2)
    model.load_state_dict(torch.load('state_dict_model.pt'))
    model.eval()

    return model

_model = load_model()

def read_image(image):
    pil_image = Image.open(BytesIO(image))
    return pil_image

def preprocess(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)

    return image

def predict(image):
    val, pred = torch.max(_model(image), 1)

    class_names=['masked', 'non_masked']

    return val.item(), class_names[pred.item()]
