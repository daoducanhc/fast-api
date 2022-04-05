import cv2
import numpy as np
import os
from glob import glob
import uuid

import torch.nn as nn
import torch.nn.functional as F

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

        return F.log_softmax(x, dim=1)

import torch
from torchvision import transforms
from PIL import Image

model = Net(2)
model.load_state_dict(torch.load('state_dict_model.pt'))
model.eval()

tol_mean = [0.5686043, 0.47916356, 0.44574347]
tol_std = [0.28832513, 0.26202822, 0.2570394 ]

transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor(),
                                transforms.Normalize(tol_mean, tol_std)
                                ])

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw faces on image
    for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                roi = frame[y:y1, x:x1]
                if roi.size != 0:
                    roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                    roi = transform(roi)
                    roi = roi.unsqueeze(0)

                    val, pred = torch.max(model(roi), 1)
                    if(pred==1):
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 225, 0), 2)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()